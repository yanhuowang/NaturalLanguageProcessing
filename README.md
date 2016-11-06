# NaturalLanguageProcessing
This repository contains 5 projects about natural language processing.

## 1. Sentiment Analysis
This project focuses on how machine learning based MaxEnt classifier works in sentiment analysis. 
 - I investigated different feature sets for training (including using unigram features only, using bigram features only, using both unigram and bigram features, adding position information and adding parts of speed tagging) in order to obtain the best results in the classification. 
 - Information gain calculation was used to prevent from overfitting and the curse of dimensionality. 
 - Negations (e.g. not, isn’t) were handled by a preprocessing technique to improve the classification performance. 
 - The results were evaluated by accuracy, precision, recall and F-measure.

## 2. Finite State Automata
Built two FSAs:
 - A deterministic FSA to recognize dates, e.g. 01/31/2015. 
 - A non-deterministic FSA, so the dates such as 1/31/2015 and 01/31/2015 could both be recognized.

## 3. Language Modeling
A bigram language model and a trigram language model were built for the Brown corpus.
 - Implemented Laplace smoothing.
 - Implemented simple linear interpolation.
 - Implement the deleted interpolation algorithm to estimate the interpolation weights using the held out corpus.
 - Calculated the perplexity of the test set.

## 4. Part of Speech Tagging with Hidden Markov models
Build a Bigram HMM part-of-speech tagger.
 - Implement the most common class algorithm as baseling algorithm.
 - Trained the data, then calculated the percent ambiguity.
 - Implemented the Viterbi method for POS tagging a test sentence.

## 5. Statistical Parsing
Built a Probabilistic Context-Free Grammars parser.
 - Used preprocessed training set, learn a PCFG using the function induce_pcfg provided by NLTK’s grammar module.
 - Implemented the probabilistic CKY algorithm for parsing a test sentence using learned PCFG.
 - Added parent annotation to improve parsing performance.
