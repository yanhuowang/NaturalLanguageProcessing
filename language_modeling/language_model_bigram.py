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

# bigram language model
class BigramLM:
    def __init__(self, vocabulary = set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.probs = {}
        self.log_probs = {}
        self.num_tokens = 0

    # 1.2.1 EstimateBigrams estimates bigram MLE probabilities given the preprocessed training data.
    def EstimateBigrams(self, training_set_prep):
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

        # calculate probs and log_probs
        for word_tuple, tuple_count in self.bigram_counts.items():
            word1 = word_tuple[0]
            word1_count = self.unigram_counts[word1]
            probability = float(tuple_count) / float(word1_count)
            self.probs[word_tuple] = probability
            self.log_probs[word_tuple] = log(probability)


    # CheckDistribution checks the validity of our bigram estimates
    def CheckDistribution(self):
        # check 0 <= prob <= 1
        for word_tuple, prob in self.probs.items():
            assert prob >= 0, "smaller than 0!"
            assert prob <= 1, "larger than 1!"
        # check the sum to 1
        for word in self.unigram_counts:
            prob_sum = 0.0;
            for word_tuple, prob in self.probs.items():
                if word_tuple[0] == word:
                    prob_sum = prob_sum + prob
            if (word != end_token):
                assert (abs(prob_sum - 1.0) <= 0.0001), "probs not sum to one!"
        print "It is a valid distribution!"

    
    # calculate the perplexity
    def Perplexity(self, test_set_prep):
        log_p_sum = 0
        n = 0
        for i in range(len(test_set_prep)):
            for j in range(len(test_set_prep[i]) - 1):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                if (word1, word2) in self.log_probs.keys():
                    log_prob = self.log_probs[(word1, word2)]
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
            for j in range(len(test_set_prep[i]) - 1):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                if (word1, word2) in self.log_probs.keys():
                    prob = (float)(self.bigram_counts[(word1, word2)] + 1) / (float)(self.unigram_counts[word1] + len(vocabulary) + 3)
                else:
                    prob = 1.0 / (float)(self.unigram_counts[word1] + len(vocabulary) + 3)
                log_prob = log(prob)
                log_p_sum = log_p_sum + log_prob
        log_pp = (-1) * log_p_sum / n
        pp = exp(log_pp)
        return pp

    # perplexity after linear interpulation
    def Perplexity_interpolation(self, test_set_prep, lambda1, lambda2):
        log_p_sum = 0
        n = 0
        for i in range(len(test_set_prep)):
            for j in range(len(test_set_prep[i]) - 1):
                n = n + 1
                word1 = test_set_prep[i][j]
                word2 = test_set_prep[i][j + 1]
                prob_bi = 0
                if (word1, word2) in self.log_probs.keys():
                    prob_bi = self.probs[(word1, word2)]
                prob_uni = (float)(self.unigram_counts[word2]) / (float)(self.num_tokens)
                prob = lambda2 * prob_bi + lambda1 * prob_uni
                log_prob = log(prob)
                log_p_sum = log_p_sum + log_prob
        log_pp = (-1) * log_p_sum / n
        pp = exp(log_pp)
        return pp

    # calculate lambda1 and lambda2
    def Estimate_interpolation_weights(self, held_out_set_prep):
        num_tokens_held_out = 0
        unigram_counts_held_out = {}
        bigram_counts_held_out = {}

        for i in range(len(held_out_set_prep)):
            for j in range(len(held_out_set_prep[i])):
                num_tokens_held_out = num_tokens_held_out + 1
                word = held_out_set_prep[i][j]

                #add to unigram_counts
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
        lambda1 = 0;
        lambda2 = 0;
        for word_tuple, tuple_count in bigram_counts_held_out.items():
            word1 = word_tuple[0]
            word2 = word_tuple[1]
            word1_count = unigram_counts_held_out[word1]
            word2_count = unigram_counts_held_out[word2]
            temp1 = (float) (word2_count - 1) / (float)(num_tokens_held_out - 1)
            temp2 = 0
            test = 0
            if word1_count != 1:
                temp2 = (float) (tuple_count - 1) / (float)(word1_count - 1)

            if temp1 >= temp2:
                lambda1 = lambda1 + tuple_count
            else:
                lambda2 = lambda2 + tuple_count   
        lambda_sum = lambda1 + lambda2
        lambda1 = (float) (lambda1) / (float) (lambda_sum)
        lambda2 = (float) (lambda2) / (float) (lambda_sum)
        return (lambda1, lambda2)

def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]

    # Text preprocessing:
    # a. set up vocabulary
    vocabulary = Set_up_volcabulary(training_set)   

    # b. Preprocess the data sets by eliminating unknown words and adding sentence boundary tokens.
    training_set_prep = PreprocessText(training_set, vocabulary)
    held_out_set_prep = PreprocessText(held_out_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    # c. Print the first sentence of each data set.    
    #print training_set_prep[0]
    #print held_out_set_prep[0]
    #print test_set_prep[0]
  
    # a. estimate a bigram_lm object
    bigramLM = BigramLM(vocabulary)
    bigramLM.EstimateBigrams(training_set_prep)

    # b. check its distribution
    # bigramLM.CheckDistribution()

    # c. compute its perplexity.
    pp = bigramLM.Perplexity(test_set_prep)
    print "The perplexity of the test corpus is:", pp

    # Laplace smoothing
    pp_laplace = bigramLM.Perplexity_laplace(test_set_prep, vocabulary)
    print "The perplexity after Laplace smoothing is:", pp_laplace

    # Linear interpolation with lambda1 = 0.5 and lambda2 = 0.5
    pp_interpolation = bigramLM.Perplexity_interpolation(test_set_prep, 0.5, 0.5)
    print "The perplexity after linear interpolation is:", pp_interpolation

    # Estimate interpolation weights
    lambdas = bigramLM.Estimate_interpolation_weights(held_out_set_prep)
    print "lambda1 is:", lambdas[0], "lambda2 is:", lambdas[1]

    pp_interpolation_new = bigramLM.Perplexity_interpolation(test_set_prep, lambdas[0], lambdas[1])
    print "perplexity after linear interpolation with the estimated interpolation weights is:", pp_interpolation_new
    
if __name__ == "__main__": 
    main()







    