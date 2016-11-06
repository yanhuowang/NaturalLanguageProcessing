import sys
import copy
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

# set up the volcabulary
def Set_up_volcabulary(training_set):
    # a. count the number of times each word appear in the training set
    word_stat = {}

    for i in range(len(training_set)):
        for j in range(len(training_set[i])):
            temp = training_set[i][j]
            if temp in word_stat:
                word_stat[temp] = word_stat[temp] + 1
            else:
                word_stat[temp] = 1
    
    # b. add the words appeared more than 1 time to the vocabulary
    vocabulary = set()

    for word, count in word_stat.items():
        if count >= 2:
            vocabulary.add(word)
    return vocabulary;

# Preprocess the data sets by eliminating unknown words and adding sentence boundary tokens.
def PreprocessText(the_set, vocabulary):
    the_set_prep  = []
    for i in range(len(the_set)):
        one_sent = []
        for j in range(len(the_set[i])):
            temp = the_set[i][j]
            if temp in vocabulary:
                one_sent.append(temp)
            else:
                one_sent.append(unknown_token)
        the_set_prep.append(one_sent)
    for i in range(len(the_set_prep)):
        the_set_prep[i].insert(0, start_token)
        the_set_prep[i].append(end_token)
    return the_set_prep

class TrigramLM:
    def __init__(self, vocabulary = set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.trigram_counts = defaultdict(float)
        self.probs_bi = {}
        self.log_probs_bi = {}
        self.probs_tri = {}
        self.log_probs_tri = {}
        self.num_tokens = 0

    def EstimateTrigrams(self, training_set_prep):
        for i in range(len(training_set_prep)):
            for j in range(len(training_set_prep[i])):
                # update num_tokens
                self.num_tokens = self.num_tokens + 1

                # add to unigram_counts
                word = training_set_prep[i][j]
                if word in self.unigram_counts:
                    self.unigram_counts[word] = self.unigram_counts[word] + 1
                else:
                    self.unigram_counts[word] = 1

                # add to bigram_counts
                if j < len(training_set_prep[i]) - 1:
                    word1 = training_set_prep[i][j]
                    word2 = training_set_prep[i][j + 1]
                    word_tuple = (word1, word2)
                    if word_tuple in self.bigram_counts:
                        self.bigram_counts[word_tuple] = self.bigram_counts[word_tuple] + 1
                    else:
                        self.bigram_counts[word_tuple] = 1

                # add to trigram_counts
                if j < len(training_set_prep[i]) - 2:
                    word1 = training_set_prep[i][j]
                    word2 = training_set_prep[i][j + 1]
                    word3 = training_set_prep[i][j + 2]
                    word_tuple = (word1, word2, word3)
                    if word_tuple in self.trigram_counts:
                        self.trigram_counts[word_tuple] = self.trigram_counts[word_tuple] + 1
                    else:
                        self.trigram_counts[word_tuple] = 1

        # calculate probs_bi and log_probs_bi
        for word_tuple, tuple_count in self.bigram_counts.items():
            word1 = word_tuple[0]
            word1_count = self.unigram_counts[word1]
            probability = float(tuple_count) / float(word1_count)
            self.probs_bi[word_tuple] = probability
            self.log_probs_bi[word_tuple] = log(probability)

        # calculate probs_tri and log_probs_tri
        for word_tuple, tuple_count in self.trigram_counts.items():
            word1 = word_tuple[0]
            word2 = word_tuple[1]
            word12_count = self.bigram_counts[(word1,word2)]
            probability = float(tuple_count) / float(word12_count)
            self.probs_tri[word_tuple] = probability
            self.log_probs_tri[word_tuple] = log(probability)

    # Perplexity
    def Perplexity(self, test_set_prep):
        log_p_sum = 0
        n = 0
        for i in range(len(test_set_prep)):
            for j in range(len(test_set_prep[i]) - 2):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                word3 = test_set_prep[i][j + 2]
                if (word1, word2, word3) in self.log_probs_tri.keys():
                    log_prob = self.log_probs_tri[(word1, word2,word3)]
                else:
                    log_prob = -float("inf")
                log_p_sum = log_p_sum + log_prob
        log_pp = (-1) * log_p_sum / n
        pp = exp(log_pp)
        return pp

    # perplexity after Laplace smooth
    def Perplexity_laplace(self, test_set_prep, vocabulary):
        log_p_sum = 0
        n = 0
        for i in range(len(test_set_prep)):
            for j in range(len(test_set_prep[i]) - 2):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                word3 = test_set_prep[i][j + 2]
                if (word1, word2, word3) in self.log_probs_tri.keys():
                    prob = (float)(self.trigram_counts[(word1, word2, word3)] + 1) / (float)(self.bigram_counts[(word1, word2)] + len(vocabulary) ＋ 3)
                else:
                    prob = 1.0 / (float)(self.bigram_counts[(word1, word2)] + len(vocabulary) ＋ 3)
                log_prob = log(prob)
                log_p_sum = log_p_sum + log_prob
        log_pp = (-1) * log_p_sum / n
        pp = exp(log_pp)
        return pp

    # perplexity after linear interpulation
    def Perplexity_interpolation(self, test_set_prep, lambda1, lambda2, lambda3):
        log_p_sum = 0
        n = 0
        for i in range(len(test_set_prep)):
            for j in range(len(test_set_prep[i]) - 2):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                word3 = test_set_prep[i][j + 2]
                prob_tri = 0
                if (word1, word2, word3) in self.log_probs_tri.keys():
                    prob_tri = self.probs_tri[(word1, word2, word3)]
                prob_bi = 0
                if (word2, word3) in self.log_probs_bi.keys():
                    prob_bi = self.probs_bi[(word2, word3)]
                prob_uni = (float)(self.unigram_counts[word3]) / (float)(self.num_tokens)
                prob = lambda3 * (float)(prob_tri) + lambda2 * (float)(prob_bi) + lambda1 * (float)(prob_uni)
                if prob == 0:
                    print word1, word2, word3, prob_uni, prob_bi, prob_tri, prob
                log_prob = log(prob)
                log_p_sum = log_p_sum + log_prob
        log_pp = (-1) * log_p_sum / n
        pp = exp(log_pp)
        return pp

    # calculate lambda1, lambda3, lambda2
    def Estimate_interpolation_weights(self, held_out_set_prep):
        num_tokens_held_out = 0
        unigram_counts_held_out = {}
        bigram_counts_held_out = {}
        trigram_counts_held_out = {}

        for i in range(len(held_out_set_prep)):
            for j in range(len(held_out_set_prep[i])):
                # update num_tokens
                num_tokens_held_out = num_tokens_held_out + 1

                # add to unigram_counts
                word = held_out_set_prep[i][j]
                if word in unigram_counts_held_out:
                    unigram_counts_held_out[word] = unigram_counts_held_out[word] + 1
                else:
                    unigram_counts_held_out[word] = 1

                # add to bigram_counts
                if j < len(held_out_set_prep[i]) - 1:
                    word1 = held_out_set_prep[i][j]
                    word2 = held_out_set_prep[i][j + 1]
                    word_tuple = (word1, word2)
                    if word_tuple in bigram_counts_held_out:
                        bigram_counts_held_out[word_tuple] = bigram_counts_held_out[word_tuple] + 1
                    else:
                        bigram_counts_held_out[word_tuple] = 1

                # add to trigram_counts
                if j < len(held_out_set_prep[i]) - 2:
                    word1 = held_out_set_prep[i][j]
                    word2 = held_out_set_prep[i][j + 1]
                    word3 = held_out_set_prep[i][j + 2]
                    word_tuple = (word1, word2, word3)
                    if word_tuple in trigram_counts_held_out:
                        trigram_counts_held_out[word_tuple] = trigram_counts_held_out[word_tuple] + 1
                    else:
                        trigram_counts_held_out[word_tuple] = 1
       
        lambda1 = 0;
        lambda2 = 0;
        lambda3 = 0;
        for word_tuple, tuple_count in trigram_counts_held_out.items():
            word1 = word_tuple[0]
            word2 = word_tuple[1]
            word3 = word_tuple[2]
            word12_count = bigram_counts_held_out[(word1, word2)]
            word23_count = bigram_counts_held_out[(word2, word3)]
            word2_count = unigram_counts_held_out[word2]
            word3_count = unigram_counts_held_out[word3]
            temp1 = 0
            temp2 = 0
            temp3 = 0
            temp1 = (float) (word3_count - 1) / (float)(num_tokens_held_out - 1)
            if word2_count != 1:
                temp2 = (float) (word23_count - 1) / (float)(word2_count - 1)
            if word12_count != 1:
                temp3 = (float) (tuple_count) / (float)(word12_count - 1)
            
            if temp1 >= temp2 and temp1 >= temp3:
                lambda1 = lambda1 + tuple_count
            elif temp2 >= temp1 and temp2 >= temp3:
                lambda2 = lambda2 + tuple_count
            else:
                lambda3 = lambda3 + tuple_count  
        lambda_sum = lambda1 + lambda2 + lambda3
        lambda1 = (float) (lambda1) / (float) (lambda_sum)
        lambda2 = (float) (lambda2) / (float) (lambda_sum)
        lambda3 = (float) (lambda3) / (float) (lambda_sum)
        return (lambda1, lambda2, lambda3)

def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]

    vocabulary = Set_up_volcabulary(training_set)   

    training_set_prep = PreprocessText(training_set, vocabulary)
    held_out_set_prep = PreprocessText(held_out_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
 
    # estimate a bigram_lm object
    trigramLM = TrigramLM(vocabulary)
    trigramLM.EstimateTrigrams(training_set_prep)

    # compute its perplexity.
    pp = trigramLM.Perplexity(test_set_prep)
    print "The perplexity of the test corpus is:", pp

    # Laplace smoothing
    pp_laplace = trigramLM.Perplexity_laplace(test_set_prep, vocabulary)
    print "The perplexity after Laplace smoothing is:", pp_laplace

    # Linear interpolation (SLI) with lambda = 1/3
    pp_interpolation = trigramLM.Perplexity_interpolation(test_set_prep, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    print "The perplexity after linear interpolation is:", pp_interpolation

    # Estimate interpolation weights
    lambdas = trigramLM.Estimate_interpolation_weights(held_out_set_prep)
    print "lambda1 is:", lambdas[0], "lambda2 is:", lambdas[1], "lambda3 is:", lambdas[2]

    pp_interpolation_new = trigramLM.Perplexity_interpolation(test_set_prep, lambdas[0], lambdas[1], lambdas[2])
    print "The perplexity after linear interpolation with the estimated interpolation weights is:", pp_interpolation_new
    
if __name__ == "__main__": 
    main()







    