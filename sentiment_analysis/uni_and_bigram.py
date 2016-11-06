import nltk
from nltk.corpus import movie_reviews
from collections import defaultdict
from nltk.metrics import BigramAssocMeasures
from math import log, exp
import random

class Sentiment:
	def __init__(self):
		# uni
		self.uni_count_pos = defaultdict(int)
		self.uni_count_neg = defaultdict(int)
		self.uni_count_total = defaultdict(int)
		self.uni_stat = defaultdict(int)
		self.num_word_pos = 0
		self.num_word_neg = 0
		self.num_word_total = 0
		self.uni_feats_total = []
		self.uni_high_score_list = []
		# bi
		self.bi_count_pos = defaultdict(int)
		self.bi_count_neg = defaultdict(int)
		self.bi_count_total = defaultdict(int)
		self.bi_stat = defaultdict(int)
		self.num_bi_pos = 0
		self.num_bi_neg = 0
		self.num_bi_total = 0
		self.bi_feats_total = []
		self.bi_high_score_list = []

	# return the uni and bi statistics e.g. how many positive words in the corpus? how many negative tuples?
	def Set_uni_bi_stat(self, training_set):
		# uni
		for i in range(len(training_set)):
			if training_set[i][1] == 'pos':
				self.num_word_pos += len(training_set[i][0])
				for j in range(len(training_set[i][0])):
					word = training_set[i][0][j]
					self.uni_count_pos[word] += 1
					self.uni_count_total[word] += 1
			elif training_set[i][1] == 'neg':
				self.num_word_neg += len(training_set[i][0])
				for j in range(len(training_set[i][0])):
					word = training_set[i][0][j]
					self.uni_count_neg[word] += 1
					self.uni_count_total[word] += 1
			else:
				print "error: neither pos nor neg"
	
		self.num_word_total = self.num_word_pos + self.num_word_neg

		# bi
		for i in range(len(training_set)):
			if training_set[i][1] == 'pos':
				self.num_bi_pos += len(training_set[i][0]) - 1
				for j in range(len(training_set[i][0]) - 1):
					word1 = training_set[i][0][j]
					word2 = training_set[i][0][j + 1]
					self.bi_count_pos[(word1, word2)] += 1
					self.bi_count_total[(word1, word2)] += 1
			elif training_set[i][1] == 'neg':
				self.num_bi_neg += len(training_set[i][0]) - 1
				for j in range(len(training_set[i][0]) - 1):
					word1 = training_set[i][0][j]
					word2 = training_set[i][0][j + 1]
					self.bi_count_neg[(word1, word2)] += 1
					self.bi_count_total[(word1, word2)] += 1
			else:
				print "error: neither pos nor neg"
		
		self.num_bi_total = self.num_bi_pos + self.num_bi_neg

	# conclude uni and bigram stat
	def Conclude_uni_bi_stat(self):
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

		# bi
		for bi, count in self.bi_count_total.items():
			if count <= 10:
				self.bi_stat[str(count)] += 1;
			elif count > 10 and count <= 15:
				self.bi_stat["10-15"] += 1;
			elif count > 15 and count <= 20:
				self.bi_stat["15-20"] += 1;
			elif count > 20 and count <= 30:
				self.bi_stat["20-30"] += 1;
			elif count > 30 and count <= 40:
				self.bi_stat["30-40"] += 1;
			elif count > 40 and count <= 50:
				self.bi_stat["40-50"] += 1;
			elif count > 50 and count <= 100:
				self.bi_stat["50-100"] += 1;
			else:
				self.bi_stat[">100"] += 1;
		print "self.bi_stat: "
		print self.bi_stat

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

	# return the word list from highest frequency to the lowest frequency
	def Uni_count_list(self):
		uni_feats_total_count = sorted(self.uni_count_total.iteritems(), key=lambda (k, v): (-v, k))
		self.uni_feats_total = [i[0] for i in uni_feats_total_count]
		print self.uni_feats_total[:10]

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
		print self.uni_high_score_list[:10]

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

	# return the tuple list from highest frequency to the lowest frequency
	def Bi_count_list(self):
		bi_feats_total_count = sorted(self.bi_count_total.iteritems(), key=lambda (k, v): (-v, k))
		self.bi_feats_total = [i[0] for i in bi_feats_total_count]
		print self.bi_feats_total[:10]

	# return the bigram list from highest score to the lowest score from info gain calculation
	def Bi_high_score_list(self):
		bi_scores = {}
 
		for bi, freq in self.bi_count_total.iteritems():
			pos_score = BigramAssocMeasures.chi_sq(self.bi_count_pos[bi],
				(freq, self.num_bi_pos), self.num_bi_total)
			neg_score = BigramAssocMeasures.chi_sq(self.bi_count_neg[bi],
				(freq, self.num_bi_neg), self.num_bi_total)
			bi_scores[bi] = pos_score + neg_score
		best = sorted(bi_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:500]
		self.bi_high_score_list = [i[0] for i in best]
		print self.bi_high_score_list[:10]

	# Feature Extractor for bigram document-list of word feature_list-list of tuples
	def Get_bi_features(self, document, feature_list): 
		document_bis = set()
		for i in range(len(document) - 1):
			t = (document[i], document[i + 1])
			document_bis.add(t)
		
		features = {}
		for bi in feature_list:
			features['contains_bi(%s)' % str(bi)] = (bi in document_bis)
		return features

	# return the feature set
	def Feature_set_bi(self, cutoff, the_set, mode): # cutoff is the num of features used
		if mode == 1: # without info gain
			bi_features = self.bi_feats_total[:cutoff]
		elif mode == 2: # with info gain
			bi_features = self.bi_high_score_list[:cutoff]
		else:
			print "error: no such mode!"
		featuresets = [(self.Get_bi_features(d, bi_features), c) for (d,c) in the_set]
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
	sentiment.Set_uni_bi_stat(training_set)
	sentiment.Conclude_uni_bi_stat()

	real_labels = sentiment.Real_labels(held_out_set)
	
	# MaxEnt with only Bigram features
	print "--- MaxEnt---"
	

	cutoffs1 = [0, 25, 50, 100, 250, 500, 1000]
	cutoffs2 = [0, 12, 25, 50, 125, 250, 500]
	sentiment.Uni_count_list()
	sentiment.Bi_count_list()

	# with info gain
	print "--- Unigram + Bigram with information gain---"
	sentiment.Uni_high_score_list()
	sentiment.Bi_high_score_list()	

	for cutoff in cutoffs2:
		print "uni features: 500"
		print "bi features: " + str(cutoff)
		
		trainsets_uni = sentiment.Feature_set_uni(500, training_set, 2)
		trainsets_bi = sentiment.Feature_set_bi(cutoff, training_set, 2)

		# combine the feature set of unigram and bigram
		trainsets_uni_bi = []
		for i in range(len(trainsets_uni)):
			temp = dict(trainsets_uni[i][0])
			temp.update(trainsets_bi[i][0])
			trainsets_uni_bi.append((temp, trainsets_uni[i][1]))

		held_out_sets_uni = sentiment.Feature_set_uni(500, held_out_set, 2)
		held_out_sets_bi = sentiment.Feature_set_bi(cutoff, held_out_set, 2)
		held_out_sets_uni_bi = []
		for i in range(len(held_out_sets_uni)):
			temp = dict(held_out_sets_uni[i][0])
			temp.update(held_out_sets_bi[i][0])
			held_out_sets_uni_bi.append((temp, held_out_sets_uni[i][1]))

		algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
		classifier_uni_bi = nltk.MaxentClassifier.train(trainsets_uni_bi, algorithm, max_iter = 50)
		classifier_uni_bi.show_most_informative_features(10)
		print "\nClassifier Accuracy : %4f\n" % nltk.classify.accuracy(classifier_uni_bi, held_out_sets_uni_bi)
		predicted_labels_uni_bi = sentiment.Maxent_predicted(classifier_uni_bi, held_out_sets_uni_bi)
		sentiment.Evaluation(predicted_labels_uni_bi, real_labels)

	# without infor grain
	print "--- Unigram + Bigram without info gain---"
	for cutoff in cutoffs1:
		print "uni features: 1000"
		print "bi features: " + str(cutoff)
		
		trainsets_uni = sentiment.Feature_set_uni(1000, training_set, 1)
		trainsets_bi = sentiment.Feature_set_bi(cutoff, training_set, 1)

		# combine the feature set of unigram and bigram
		trainsets_uni_bi = []
		for i in range(len(trainsets_uni)):
			temp = dict(trainsets_uni[i][0])
			temp.update(trainsets_bi[i][0])
			trainsets_uni_bi.append((temp, trainsets_uni[i][1]))
		held_out_sets_uni = sentiment.Feature_set_uni(1000, held_out_set, 1)
		held_out_sets_bi = sentiment.Feature_set_bi(cutoff, held_out_set, 1)
		held_out_sets_uni_bi = []
		for i in range(len(held_out_sets_uni)):
			temp = dict(held_out_sets_uni[i][0])
			temp.update(held_out_sets_bi[i][0])
			held_out_sets_uni_bi.append((temp, held_out_sets_uni[i][1]))

		algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
		classifier_uni_bi = nltk.MaxentClassifier.train(trainsets_uni_bi, algorithm, max_iter = 50)
		classifier_uni_bi.show_most_informative_features(10)
		print "\nClassifier Accuracy : %4f\n" % nltk.classify.accuracy(classifier_uni_bi, held_out_sets_uni_bi)
		predicted_labels_uni_bi = sentiment.Maxent_predicted(classifier_uni_bi, held_out_sets_uni_bi)
		sentiment.Evaluation(predicted_labels_uni_bi, real_labels)
	

if __name__ == "__main__": 
	main()