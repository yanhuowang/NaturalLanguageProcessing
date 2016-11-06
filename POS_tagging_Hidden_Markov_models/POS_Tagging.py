import sys
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

# Remove trace tokens and tags from the treebank as these are not necessary.
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

# set up the volcabulary
def Set_up_volcabulary(training_set):
    # a. count the number of times each word appear in the training set
    word_stat = defaultdict(int)
    for i in range(len(training_set)):
        for j in range(len(training_set[i])):
            temp = training_set[i][j]
            word_stat[temp[0]] += 1;
    
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
            if temp[0] in vocabulary:
                one_sent.append(temp)
            else:
                one_sent.append((unknown_token, temp[1]))
        the_set_prep.append(one_sent)
    for i in range(len(the_set_prep)):
        the_set_prep[i].insert(0, (start_token, start_token))
        the_set_prep[i].append((end_token, end_token))
    return the_set_prep

# Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline.      
def MostCommonClassBaseline(training_set, test_set):
    word_tag_stat = {}
    for i in range(len(training_set)):
        for j in range(len(training_set[i])):
            word = training_set[i][j][0]
            tag = training_set[i][j][1]
            word_tag_stat[word] = word_tag_stat.get(word, {})
            word_tag_stat[word][tag] = word_tag_stat[word].get(tag, 0) + 1

    test_set_predicted_baseline = []
    for i in range(len(test_set)):
        one_sent = []
        for j in range(len(test_set[i])):
            word = test_set[i][j][0]
            tag = max(word_tag_stat[word], key = word_tag_stat[word].get)
            one_sent.append((word, tag))
        test_set_predicted_baseline.append(one_sent)
    return test_set_predicted_baseline


# Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted.  
def ComputeAccuracy(test_set, test_set_predicted):
    word_count = 0
    sentence_count = 0
    correct_word_count = 0
    correct_sentence_count = 0

    for i in range(len(test_set_predicted)):
        if test_set_predicted[i] == []: # skip the sentences viterbi got zero.
            continue;
        sentence_count += 1
        all_correct = True
        for j in range(1, len(test_set_predicted[i]) - 1):
            word_count += 1
            if test_set_predicted[i][j][1] == test_set[i][j][1]:
                correct_word_count += 1
            else:
                all_correct = False
        if all_correct:
            correct_sentence_count += 1

    print "sentence_accuracy:", float(correct_sentence_count) / float(sentence_count)
    print "word_accuracy:", float(correct_word_count) / float(word_count)

# Confusion Matrix   
def ConfusionMatrix (test_set, test_set_predicted):
    confusion_counts = {}
    confusion_percentage = {}
    top_ten_confusion_counts = {}
    top_ten_confusion_percentage = {}

    for i in range(len(test_set_predicted)):
        if test_set_predicted[i] == []: # skip the sentences viterbi got zero.
            continue;
        for j in range(1, len(test_set_predicted[i]) - 1):
            correct_tag = test_set[i][j][1]
            predict_tag = test_set_predicted[i][j][1]
            if predict_tag != correct_tag:
                confusion_counts[(correct_tag, predict_tag)] = confusion_counts.get((correct_tag, predict_tag), 0) + 1
                            
    top_ten_confusion_counts = sorted(confusion_counts.iteritems(), key=lambda (k, v): (-v, k))[:10]
    print "10 most common confusion pairs with counts:", top_ten_confusion_counts

    # calculate the percentage
    factor = 100.0 / sum(confusion_counts.itervalues())
    for k in confusion_counts:
        confusion_percentage[k] = confusion_counts[k] * factor
    top_ten_confusion_percentage = sorted(confusion_percentage.iteritems(), key=lambda (k, v): (-v, k))[:10]
    print "10 most common confusion pairs with percentage: ", top_ten_confusion_percentage

class BigramHMM:
    def __init__(self):
        # A matrix: a_{ij} = P(t_j | t_i)
        self.transitions = {}
        self.log_transitions = {}
        # B matrix: b_{ii} = P(w_i | t_i)
        self.emissions = {}
        self.log_emissions = {}
        # a dictionary that maps a word to the set of possible tags
        self.dictionary = {}

        self.num_tags = 0
        self.tag_unigram_counts = defaultdict(int)
        self.tag_bigram_counts = defaultdict(int)
        self.word_tag_counts = defaultdict(int)
    
    def Train(self, training_set):
        # calculate helper variables
        for i in range(len(training_set)):
            for j in range(len(training_set[i])):
                self.num_tags = self.num_tags + 1

                # add to tag_unigram_counts and add to word_tag_counts
                word = training_set[i][j][0]
                tag = training_set[i][j][1]
                self.tag_unigram_counts[tag] = self.tag_unigram_counts.get(tag, 0) + 1
                self.word_tag_counts[(word, tag)] = self.word_tag_counts.get((word, tag), 0) + 1

                # add to tag_bigram_counts
                if j < len(training_set[i]) - 1:
                    tag1 = training_set[i][j][1]
                    tag2 = training_set[i][j + 1][1]
                    self.tag_bigram_counts[(tag1, tag2)] = self.tag_bigram_counts.get((tag1, tag2), 0) + 1
        # A matrix: {(tag1, tag2): p, ...}
        for tag_tuple, tuple_count in self.tag_bigram_counts.items():
            tag1 = tag_tuple[0]
            tag1_count = self.tag_unigram_counts[tag1]
            probability = float(tuple_count) / float(tag1_count)
            self.transitions[tag_tuple] = probability
            self.log_transitions[tag_tuple] = log(probability)                    
        # B matrix: {(word, tag): p, ...}
        for word_tag_tuple, tuple_count in self.word_tag_counts.items():
            tag = word_tag_tuple[1]
            tag_count = self.tag_unigram_counts[tag]
            probability = float(tuple_count) / float(tag_count)
            self.emissions[word_tag_tuple] = probability
            self.log_emissions[word_tag_tuple] = log(probability)
        # tag dictionary {word, set(tag1, tag2...)}
        for i in range(len(training_set)):
            for j in range(len(training_set[i])):
                word = training_set[i][j][0]
                tag = training_set[i][j][1]
                self.dictionary[word] = self.dictionary.get(word, set())
                self.dictionary[word].add(tag)

    # Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary.             
    def ComputePercentAmbiguous(self, data_set):
        num_tokens = 0
        num_tokens_ambig = 0
        for i in range(len(data_set)):
            for j in range(len(data_set[i])):
                word = data_set[i][j][0]
                num_tokens += 1
                if len(self.dictionary[word]) > 1:
                    num_tokens_ambig += 1
        percent_ambig = float(num_tokens_ambig) / float(num_tokens)
        
        print "Number of tags for the unknown tokens is:", len(self.dictionary[unknown_token])
        print "They are:", self.dictionary[unknown_token]
        return 100.0 * percent_ambig

    # Compute the percentage of words in data_set that have more than one tag according to self.dictionary.
    def ComputePercentAmbiguous1(self, vocabulary):
        num_tokens = 0
        num_tokens_ambig = 0
        for word in vocabulary:
            num_tokens += 1
            if len(self.dictionary[word]) > 1:
                num_tokens_ambig += 1
        num_tokens += 3 # <S>, </S> and <UNK>
        num_tokens_ambig += 1 # <UNK>
        percent_ambig = float(num_tokens_ambig) / float(num_tokens)
        return 100.0 * percent_ambig

    # Compute the joint probability of the words and tags of a tagged sentence. 
    def JointProbability(self, sent):
        log_joint_p = 0.0
        for i in range(len(sent)):
            word = sent[i][0]
            tag = sent[i][1]
            log_joint_p += self.log_emissions[(word,tag)]
            if i < len(sent) - 1:
                tag_next = sent[i + 1][1]
                log_joint_p += self.log_transitions[(tag,tag_next)]
        return exp(log_joint_p)

    # Find the probability and identity of the most likely tag sequence given the sentence.              
    def Viterbi(self, sent):
        # viterbi is a list of dictionary that maps each tag t to the probablity of the best tag sequence that ends in t
        viterbi = []
        # backpointer is a list of dictionary that maps each tag t to the previous tag in the best tag sequence
        backpointer = []

        first_viterbi = {}
        first_backpointer = {}
        for tag in self.dictionary[sent[1][0]]:
            if tag == start_token: 
                continue
            # we start with the first meaning for word in the sentence, since all the sentence start with <S>
            first_viterbi[tag] = self.log_transitions.get((start_token, tag), -float("inf")) + self.log_emissions.get((sent[1][0], tag), -float("inf"))
            first_backpointer[tag] = start_token
        viterbi.append(first_viterbi)
        backpointer.append(first_backpointer)

        for word_index in range(2, len(sent) - 1):
            this_viterbi = {}
            this_backpointer = {}
            prev_viterbi = viterbi[-1]
            for cur_tag in self.dictionary[sent[word_index][0]]:
                if cur_tag == start_token:
                    continue
                best_pre_tag = None
                best_prob = -float("inf")
                for pre_tag in self.dictionary[sent[word_index - 1][0]]:
                    this_prob = prev_viterbi.get(pre_tag, -float("inf")) + self.log_transitions.get((pre_tag, cur_tag), -float("inf"))
                    if this_prob > best_prob:
                        best_pre_tag = pre_tag
                        best_prob = this_prob
                this_viterbi[cur_tag] = prev_viterbi.get(best_pre_tag, -float("inf")) + self.log_transitions.get((best_pre_tag, cur_tag), -float("inf")) + self.log_emissions.get((sent[word_index][0], cur_tag), -float("inf"))
                this_backpointer[cur_tag] = best_pre_tag
            viterbi.append(this_viterbi)
            backpointer.append(this_backpointer)
        # Done with all meaningful word in the sentence. Now caculate the prob fo each tag followd by </S>
        prev_viterbi = viterbi[-1]
        best_pre_tag = None
        best_prob = -float("inf")
        for pre_tag in self.dictionary[sent[-2][0]]:
            this_prob = prev_viterbi.get(pre_tag, -float("inf")) + self.log_transitions.get((pre_tag, end_token), -float("inf"))
            if this_prob > best_prob:
                best_pre_tag = pre_tag
                best_prob = this_prob
        log_prob_tag_seq = prev_viterbi.get(best_pre_tag, -float("inf")) + self.log_transitions.get((best_pre_tag, end_token), -float("inf"))
        prob_tag_seq = exp(log_prob_tag_seq)

        # Get the best tag sequence
        if prob_tag_seq == 0.0:
            return [] # All branches is 0, so I skip this sentence
        else:
            best_tag_seq = [end_token, best_pre_tag]
            backpointer.reverse()
            cur_best_tag = best_pre_tag
            for bp in backpointer:
                best_tag_seq.append(bp[cur_best_tag])
                cur_best_tag = bp[cur_best_tag]
            best_tag_seq.reverse()
            sent_predicted = zip(untag(sent), best_tag_seq)
            return sent_predicted

    # Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set.  
    def Test(self, test_set):
        test_set_predicted = []
        for i in range(len(test_set)):
            test_set_predicted.append(self.Viterbi(test_set[i]))
        return test_set_predicted
    
def main():
    treebank_tagged_sents = TreebankNoTraces() 
    training_set = treebank_tagged_sents[:3000] 
    test_set = treebank_tagged_sents[3000:]

    # 1. Preprocessing:
    vocabulary = Set_up_volcabulary(training_set)
    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    # Print the first sentence of each data set.
    print "--- First sentence of each data set after preprocessing ---"
    print " ".join(untag(training_set_prep[0])) 
    print " ".join(untag(test_set_prep[0]))
    print "\n"

    # 2. Implement the most common class baseline. Report accuracy of the predicted tags.
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)
    print "\n"
    

    # Estimate Bigram HMM from the training set, report level of ambiguity.
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)
    print "--- Training ---"
    print "Percent tag ambiguity(tokens) in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    print "For comparison, percent tag ambiguity(words) in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous1(vocabulary)
    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])
    print "\n"

    # Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Testing ---"
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)
    print "\n"

    # confusion matrix
    print "--- Confusion matrix ---"
    ConfusionMatrix(test_set_prep, test_set_predicted_bigram_hmm) 

if __name__ == "__main__": 
    main()
    