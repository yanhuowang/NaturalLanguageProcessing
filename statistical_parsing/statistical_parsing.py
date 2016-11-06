import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow, log

unknown_token = "<UNK>"

# Removes all function tags e.g., turns NP-SBJ into NP.       
def RemoveFunctionTags(tree):
	for subtree in tree.subtrees():  # for all nodes of the tree
		# if it's a preterminal node with the label "-NONE-", then skip for now
		if subtree.height() == 2 and subtree.label() == "-NONE-": continue
		nt = subtree.label()  # get the nonterminal that labels the node
		labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
		if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
			subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"

# Return true if node is a trace node.         
def IsTraceNode(node):
	# return true if the node is a preterminal node and has the label "-NONE-"
	return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"

# Deletes any trace node children and returns true if all children were deleted.
def RemoveTraces(node):
	if node.height() == 2:  # if the node is a preterminal node
		return False  # already a preterminal, cannot have a trace node child.
	i = 0
	while i < len(node):  # iterate over the children, node[i]
		# if the child is a trace node or it is a node whose children were deleted
		if IsTraceNode(node[i]) or RemoveTraces(node[i]): 
			del node[i]  # then delete the child
		else: i += 1
	return len(node) == 0  # return true if all children were deleted
	
# Preprocessing of the Penn treebank.
def TreebankNoTraces():
	tb = []
	for t in treebank.parsed_sents():
		if t.label() != "S": continue
		RemoveFunctionTags(t)
		RemoveTraces(t)
		t.collapse_unary(collapsePOS = True, collapseRoot = True)
		t.chomsky_normal_form()
		tb.append(t)
	return tb
		
# Enumerate all preterminal nodes of the tree.
def PreterminalNodes(tree):
	for subtree in tree.subtrees():
		if subtree.height() == 2:
			yield subtree
	
# Print the tree in one line no matter how big it is e.g., (VP (VB Book) (NP (DT that) (NN flight)))        
def PrintTree(tree):
	if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
	return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))
 
# set up the volcabulary
def Set_up_volcabulary(training_set):
	# a. count the number of times each word appear in the training set
	word_stat = defaultdict(int)
	for tree in training_set:
		for node in PreterminalNodes(tree):
			word = node[0]
			word_stat[word] += 1
	# b. add the words appeared more than 1 time to the vocabulary
	vocabulary = set()
	for word, count in word_stat.items():
		if count >= 2:
			vocabulary.add(word)
	return vocabulary

# Preprocess the data sets by eliminating unknown words and adding sentence boundary tokens.
def PreprocessText(the_set, vocabulary):
	for tree in the_set:
		for node in PreterminalNodes(tree):
			if node[0] not in vocabulary:
				node[0] = unknown_token    
	return the_set

# a helper function to build a tree
def BuildTreeHelper(cky_table, sent, i, j, a):
	if i == j:
		return Tree(a.symbol(), [sent[i]])
	else:
		k = cky_table[i][j][a][0]
		b = cky_table[i][j][a][1]
		c = cky_table[i][j][a][2]
		return Tree(a.symbol(),[BuildTreeHelper(cky_table, sent, i, k, b), BuildTreeHelper(cky_table, sent, k+1, j, c)])

class InvertedGrammar:
	def __init__(self, pcfg):
		self._pcfg = pcfg
		self._r2l = defaultdict(list)
		self._r2l_lex = defaultdict(list) 
		self.BuildIndex()
		#self.PrintIndex("index") 

	def PrintIndex(self, filename):
		f = open(filename, "w")
		for rhs, prods in self._r2l.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		for rhs, prods in self._r2l_lex.iteritems():
			f.write("%s\n" %str(rhs))
			for prod in prods:
				f.write("\t%s\n" %str(prod))
			f.write("---\n")
		f.close()
	
	# Build an inverted index of your grammar that maps right hand sides of all productions to their left hands sides.
	def BuildIndex(self):
		for production in self._pcfg.productions():
			if production.is_lexical():
				self._r2l_lex[production._rhs[0]].append(production)
			else:
				self._r2l[production._rhs].append(production)
	
	# Implement the CKY algorithm for PCFGs	
	def Parse(self, sent):
		n = len(sent)
		table = [[ defaultdict(lambda: float('-inf')) for i in range(n)] for j in range(n)]
		back_pointer = [[ {} for i in range(n)] for j in range(n)]

		for j in range(0, n):
			word = sent[j]

			for production in self._r2l_lex[word]:
				tag = production._lhs
				log_prob = log(production.prob())
				table[j][j][tag] = log_prob
			
			for i in range(j-1, -1, -1):
				for k in range(i, j):
					sub1 = table[i][k]
					sub2 = table[k+1][j]
					for b_tag, b_log_prob in sub1.iteritems():
						for c_tag, c_log_prob in sub2.iteritems():
							prods_to_bc = self._r2l[(b_tag,c_tag)]
							for production in prods_to_bc:
								a_tag = production._lhs
								a_log_prob = log(production.prob()) + b_log_prob + c_log_prob
								if a_log_prob > table[i][j][a_tag]:
									table[i][j][a_tag] = a_log_prob
									back_pointer[i][j][a_tag] = (k, b_tag, c_tag)

		return (table, back_pointer)

	@staticmethod
	def BuildTree(cky_table, sent):
		n = len(sent)
		if Nonterminal("S") not in cky_table[0][n-1].keys():
			# print "not start with S"
			return None
		else:
			tree = BuildTreeHelper(cky_table, sent, 0, n-1, Nonterminal("S"))
			return tree



def main():
	treebank_parsed_sents = TreebankNoTraces()
	training_set = treebank_parsed_sents[:3000]
	test_set = treebank_parsed_sents[3000:]
	
	# Preprocessing: Transform the data sets by eliminating unknown words.
	print "--- Preprocessing ---"
	print "Before transformation, training_set is:\n"
	print PrintTree(training_set[0]) + "\n"
	print "Before transformation, test_set is:\n"
	print PrintTree(test_set[0]) + "\n"
	
	vocabulary = Set_up_volcabulary(training_set)

	training_set_prep = PreprocessText(training_set, vocabulary)
	test_set_prep = PreprocessText(test_set, vocabulary)
	print "After transformation, training_set_prep is:\n"
	print PrintTree(training_set_prep[0]) + "\n"
	print "After transformation, test_set_prep is:\n"
	print PrintTree(test_set_prep[0]) + "\n"
	

	# Training
	productions = []

	for tree in training_set_prep:
		productions += tree.productions()

	S = Nonterminal("S")
	grammar = induce_pcfg(S, productions)

	NPlist = []
	for production in grammar.productions():
		if (production._lhs.__eq__(Nonterminal("NP"))):
			NPlist.append(production)

	print "--- Training ---"
	print "Number of productions for the NP nonterminal is:" + str(len(NPlist)) + "\n"
	NPlist.sort(key=lambda x: x.prob(), reverse=True)
	print "The most probable 10 productions for the NP nonterminal is:"
	print NPlist[:10]
	print "\n"

	
	print "--- Test ---"
	
	invertedGrammar = InvertedGrammar(grammar)
	test_sentence = "Terms were n't disclosed .".split()
	parse_result = invertedGrammar.Parse(test_sentence)
	table = parse_result[0]
	back_pointer = parse_result[1]
	print "3.2 log probability:" + str(table[0][-1][Nonterminal("S")])
	print "\n"

	tree = invertedGrammar.BuildTree(back_pointer, test_sentence)
	print "3.3 the parse tree is:"
	print tree
	print "\n"
	print PrintTree(tree)
	print "\n"

	bucket1 = []
	bucket2 = []
	bucket3 = []
	bucket4 = []
	bucket5 = []
	#bucket_test = []

	for tree in test_set_prep:
		sent_len = len(tree.leaves())
		if sent_len > 0 and sent_len < 10:
			bucket1.append(tree)
			#bucket_test.append(tree)
		elif sent_len >= 10 and sent_len < 20:
			bucket2.append(tree)
			#bucket_test.append(tree)
		elif sent_len >= 20 and sent_len < 30:
			bucket3.append(tree)
		elif sent_len >= 30 and sent_len < 40:
			bucket4.append(tree)
		elif sent_len >= 40:
			bucket5.append(tree)
		else:
			print "error"

	print str(len(bucket1)) + " sentences fall in bucket1"
	print str(len(bucket2)) + " sentences fall in bucket2"
	print str(len(bucket3)) + " sentences fall in bucket3"
	print str(len(bucket4)) + " sentences fall in bucket4"
	print str(len(bucket5)) + " sentences fall in bucket5"

	# Sanity test
	f_test = open("test", "w")
	f_gold = open("gold", "w")
	for tree in bucket_test:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test.write(PrintTree(tree_predicted))
			f_test.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold.write(PrintTree(tree_gold))
		f_gold.write("\n")

	f_test.close()
	f_gold.close()

	# Bucket1
	f_test_1 = open("test_1", "w")
	f_gold_1 = open("gold_1", "w")
	for tree in bucket1:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_1.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_1.write(PrintTree(tree_predicted))
			f_test_1.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_1.write(PrintTree(tree_gold))
		f_gold_1.write("\n")

	f_test_1.close()
	f_gold_1.close()

	# Bucket2
	f_test_2 = open("test_2", "w")
	f_gold_2 = open("gold_2", "w")
	for tree in bucket2:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_2.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_2.write(PrintTree(tree_predicted))
			f_test_2.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_2.write(PrintTree(tree_gold))
		f_gold_2.write("\n")

	f_test_2.close()
	f_gold_2.close()

	# Bucket3
	f_test_3 = open("test_3", "w")
	f_gold_3 = open("gold_3", "w")
	for tree in bucket3:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_3.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_3.write(PrintTree(tree_predicted))
			f_test_3.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_3.write(PrintTree(tree_gold))
		f_gold_3.write("\n")

	f_test_3.close()
	f_gold_3.close()

	# Bucket4
	f_test_4 = open("test_4", "w")
	f_gold_4 = open("gold_4", "w")
	for tree in bucket4:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_4.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_4.write(PrintTree(tree_predicted))
			f_test_4.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_4.write(PrintTree(tree_gold))
		f_gold_4.write("\n")

	f_test_4.close()
	f_gold_4.close()

	# Bucket5
	f_test_5 = open("test_5", "w")
	f_gold_5 = open("gold_5", "w")
	for tree in bucket5:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_5.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_5.write(PrintTree(tree_predicted))
			f_test_5.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_5.write(PrintTree(tree_gold))
		f_gold_5.write("\n")

	f_test_5.close()
	f_gold_5.close()
	
	# Overall test set
	f_test_overall = open("test_overall", "w")
	f_gold_overall = open("gold_overall", "w")
	for tree in test_set_prep:	
		sent = tree.leaves()
		parse_result = invertedGrammar.Parse(sent)
		back_pointer = parse_result[1]
		tree_predicted = invertedGrammar.BuildTree(back_pointer, sent)

		if tree_predicted == None:
			f_test_overall.write("\n")
		else:
			tree_predicted.un_chomsky_normal_form()
			f_test_overall.write(PrintTree(tree_predicted))
			f_test_overall.write("\n")

		tree_gold = tree
		tree_gold.un_chomsky_normal_form()
		f_gold_overall.write(PrintTree(tree_gold))
		f_gold_overall.write("\n")

	f_test_overall.close()
	f_gold_overall.close()

if __name__ == "__main__": 
	main()  
	





