import nltk
from nltk.corpus import movie_reviews
from collections import defaultdict
from nltk.metrics import BigramAssocMeasures
from math import log, exp
import random

class Sentiment:
	def __init__(self):
		self.uni_count_pos = defaultdict(int)
		self.uni_count_neg = defaultdict(int)
		self.uni_count_total = defaultdict(int)
		self.uni_stat = defaultdict(int)
		self.num_vocabulary = 0
		self.num_sent_pos = 0
		self.num_sent_neg = 0
		self.num_word_pos = 0
		self.num_word_neg = 0
		self.num_word_total = 0
		self.uni_feats_total = []
		self.uni_high_score_list = []

	# return the word statistics e.g. how many positive words in the corpus?
	def Set_word_stat(self, training_set):
		for i in range(len(training_set)):
			if training_set[i][1] == 'pos':
				self.num_word_pos += len(training_set[i][0])
				self.num_sent_pos += 1
				for j in range(len(training_set[i][0])):
					word = training_set[i][0][j]
					self.uni_count_pos[word] += 1
					self.uni_count_total[word] += 1
			elif training_set[i][1] == 'neg':
				self.num_word_neg += len(training_set[i][0])
				self.num_sent_neg += 1
				for j in range(len(training_set[i][0])):
					word = training_set[i][0][j]
					self.uni_count_neg[word] += 1
					self.uni_count_total[word] += 1
			else:
				print "error: neither pos nor neg"
		
		self.num_vocabulary = len(self.uni_count_total)
		self.num_word_total = self.num_word_pos + self.num_word_neg

	# conclude unigram stat
	def Conclude_word_stat(self):
		# uni
		for word, count in self.uni_count_total.items():
			if count <= 10:
				self.uni_stat[str(count)] += 1;
			elif count > 10 and count <= 15:
				self.uni_stat["10-15"] += 1;
			elif count > 15 and count <= 20:
				self.uni_stat["15-20"] += 1;
			elif count > 20 and count <= 30:
				self.uni_stat["20-30"] += 1;
			elif count > 30 and count <= 40:
				self.uni_stat["30-40"] += 1;
			elif count > 40 and count <= 50:
				self.uni_stat["40-50"] += 1;
			elif count > 50 and count <= 100:
				self.uni_stat["50-100"] += 1;
			else:
				self.uni_stat[">100"] += 1;
		print "self.uni_stat: "
		print self.uni_stat

	# return the real label
	def Real_labels(self, the_set):
		real_labels = []
		for i in range(len(the_set)):
			label = the_set[i][1]
			real_labels.append(label)
		return real_labels

	# calculate the accuracy, pos_precision, neg_precision, pos_recall, neg_recall, pos_F_measure, neg_F_measure
	def Evaluation(self, predicted_labels, real_labels):
		if len(predicted_labels) != len(real_labels):
			print "error: lenth of predicted_labels differs from lenth of real_labels!"
		pos_pos = 0 # predicted positive and real positive
		pos_neg = 0
		neg_pos = 0
		neg_neg = 0
		for i in range(len(predicted_labels)):
			if predicted_labels[i] == "pos" and real_labels[i] == "pos":
				pos_pos += 1
			elif predicted_labels[i] == "pos" and real_labels[i] == "neg":
				pos_neg += 1
			elif predicted_labels[i] == "neg" and real_labels[i] == "pos":
				neg_pos += 1
			elif predicted_labels[i] == "neg" and real_labels[i] == "neg":
				neg_neg += 1
			else:
				print "error: in evaluation!"
		accuracy = float(pos_pos + neg_neg) / float(len(predicted_labels))
		if (pos_pos + pos_neg) != 0:
			pos_precision = float(pos_pos) / float(pos_pos + pos_neg)
		else:
			pos_precision = 0
		if (neg_neg + neg_pos) != 0:
			neg_precision = float(neg_neg) / float(neg_neg + neg_pos)
		else:
			neg_precision = 0
		if (pos_pos + neg_pos) != 0:
			pos_recall = float(pos_pos) / float(pos_pos + neg_pos)
		else:
			pos_recall = 0
		if (neg_neg + pos_neg) != 0:
			neg_recall = float(neg_neg) / float(neg_neg + pos_neg)
		else:
			neg_recall = 0
		if (pos_precision + pos_recall) != 0:
			pos_F_measure = 2 * ((pos_precision * pos_recall) / (pos_precision + pos_recall))
		else:
			pos_F_measure = 0
		if (neg_precision + neg_recall) != 0:
			neg_F_measure = 2 * ((neg_precision * neg_recall) / (neg_precision + neg_recall))
		else: 
			neg_F_measure = 0

		print "accuracy: " + str(accuracy)
		print "pos_precision: " + str(pos_precision)
		print "neg_precision: " + str(neg_precision)
		print "pos_recall: " + str(pos_recall)
		print "neg_recall: " + str(neg_recall)
		print "pos_F_measure: " + str(pos_F_measure)
		print "neg_F_measure: " + str(neg_F_measure)

	# return the predited labels by baseline (naive Bayes + Laplace smoothing)
	def Baseline(self, test_set):
		log_prior_pos = log(float(self.num_sent_pos) / (float(self.num_sent_pos) + float(self.num_sent_neg)))
		log_prior_neg = log(float(self.num_sent_neg) / (float(self.num_sent_pos) + float(self.num_sent_neg)))
		baseline_predicted = []
		for i in range(len(test_set)):
			logp_pos = 0
			logp_neg = 0
			for j in range(len(test_set[i][0])):
				word = test_set[i][0][j]
				p_wc_pos = float(self.uni_count_pos[word] + 1) / float(self.num_word_pos + self.num_vocabulary)
				p_wc_neg = float(self.uni_count_neg[word] + 1) / float(self.num_word_neg + self.num_vocabulary)
				logp_pos += log(p_wc_pos)
				logp_neg += log(p_wc_neg)
			logp_pos += log_prior_pos
			logp_neg += log_prior_neg
			if (logp_pos > logp_neg):
				baseline_predicted.append("pos".decode('utf-8'))
			else:
				baseline_predicted.append("neg".decode('utf-8'))
		return baseline_predicted

	# return the word list from highest frequency to the lowest frequency
	def Uni_count_list(self):
		uni_feats_total_count = sorted(self.uni_count_total.iteritems(), key=lambda (k, v): (-v, k))
		self.uni_feats_total = [i[0] for i in uni_feats_total_count]
		# print self.uni_feats_total[:10]

	# return the word list from highest score to the lowest score from info gain calculation
	def Uni_high_score_list(self):
		word_scores = {}
 
		for word, freq in self.uni_count_total.iteritems():
			pos_score = BigramAssocMeasures.chi_sq(self.uni_count_pos[word],
				(freq, self.num_word_pos), self.num_word_total)
			neg_score = BigramAssocMeasures.chi_sq(self.uni_count_neg[word],
				(freq, self.num_word_neg), self.num_word_total)
			word_scores[word] = pos_score + neg_score
		best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
		self.uni_high_score_list = [i[0] for i in best]
		# print self.uni_high_score_list[:10]

	# Feature Extractor for unigram
	def Get_uni_features(self, document, feature_list): 
		document_words = set(document)
		features = {}
		for word in feature_list:
			features['contains_uni(%s)' % word] = (word in document_words)
		return features

	# return the feature set
	def Feature_set_uni(self, cutoff, the_set, mode): # cutoff is the num of features used
		if mode == 1: # without info gain
			uni_features = self.uni_feats_total[:cutoff]
		elif mode == 2: # with info gain
			uni_features = self.uni_high_score_list[:cutoff]
		else:
			print "error: no such mode!"
		featuresets = [(self.Get_uni_features(d, uni_features), c) for (d,c) in the_set]
		return featuresets

	# return the predited labels by main method (MaxEnt)
	def Maxent_predicted(self, classifier, the_sets):
		maxent_predicted = []
		for the_set in the_sets:
			maxent_predicted.append(classifier.classify(the_set[0]))
		return maxent_predicted	


def main():
	# - categories 'neg' and 'pos'
	# - fileid ex - 'neg/cv000_29416.txt'
	documents = [(list(movie_reviews.words(fileid)), category)
		for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

	random.shuffle(documents)

	training_set = documents[:1600] #1600 sentences in training set
	held_out_set = documents[-400:-200] #200 sentences in held out set
	test_set = documents[-200:] # 200 sentences in test set

	sentiment = Sentiment()
	sentiment.Set_word_stat(training_set)
	sentiment.Conclude_word_stat()

	real_labels = sentiment.Real_labels(held_out_set)

	# Baseline
	predicted_labels_baseline = sentiment.Baseline(test_set)
	print "--- Baseline ---"
	sentiment.Evaluation(predicted_labels_baseline, real_labels)
	
	
	# MaxEnt with only unigram features
	print "--- MaxEnt ---"
	print "--- Unigram only ---"

	cutoffs = [100, 200, 500, 1000, 2000, 3000, 5000, 10000]

	# without info gain
	sentiment.Uni_count_list()
	
	for cutoff in cutoffs:
		print "cutoff = " + str(cutoff)
		trainsets_uni = sentiment.Feature_set_uni(cutoff, training_set, 1)
		held_out_sets_uni = sentiment.Feature_set_uni(cutoff, held_out_set, 1)
		algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
		classifier_uni = nltk.MaxentClassifier.train(trainsets_uni, algorithm, max_iter = 50)
		classifier_uni.show_most_informative_features(10)
		print "\nClassifier Accuracy : %4f\n" % nltk.classify.accuracy(classifier_uni, held_out_sets_uni)
		predicted_labels_uni = sentiment.Maxent_predicted(classifier_uni, held_out_sets_uni)
		sentiment.Evaluation(predicted_labels_uni, real_labels)
	
	# without info gain
	print "--- Unigram only with information gain---"
	
	sentiment.Uni_high_score_list()

	for cutoff in cutoffs:
		print "cutoff = " + str(cutoff)
		trainsets_uni = sentiment.Feature_set_uni(cutoff, training_set, 2)
		held_out_sets_uni = sentiment.Feature_set_uni(cutoff, held_out_set, 2)
		algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
		classifier_uni = nltk.MaxentClassifier.train(trainsets_uni, algorithm, max_iter = 50)
		classifier_uni.show_most_informative_features(10)
		print "\nClassifier Accuracy : %4f\n" % nltk.classify.accuracy(classifier_uni, held_out_sets_uni)
		predicted_labels_uni = sentiment.Maxent_predicted(classifier_uni, held_out_sets_uni)
		sentiment.Evaluation(predicted_labels_uni, real_labels)

if __name__ == "__main__": 
	main()