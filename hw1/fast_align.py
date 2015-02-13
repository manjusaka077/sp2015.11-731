#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import random
import math

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1...\n")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
t = defaultdict(lambda : defaultdict(float))
q = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
h = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
b = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
p0 = 0.1
l = 0.1

# initialize counters
c_ef_pair = defaultdict(lambda : defaultdict(float));
c_e_all = defaultdict(float);
#c_len_pair = defaultdict(lambda : defaultdict(lambda : defaultdict(float)));
#c_len_all = defaultdict(float);

#Preprocess data
for line_pair in bitext :
	for line in line_pair :
		for i in xrange(len(line)) :
			line[i] = line[i].lower()
	line_pair[1].insert(0, "NULL")


for line_pair in bitext : 
	f_len = len(line_pair[0])
	e_len = len(line_pair[1])
	# for each length pair, maintain a 2 dim array
	lm_pair = str(f_len) + "," + str(e_len)
	# f sentence
	for i in xrange(f_len) :

		f_word = line_pair[0][i]

		# e sentence
		for j in xrange(e_len) :
			e_word = line_pair[1][j]

			# initialize t and q from random value
			q[lm_pair][str(i)][str(j)] = 1 / (1 + float(e_len))
			h[lm_pair][str(i)][str(j)] = 0.1
			t[f_word][e_word] = 0.1

# EM algorithm
num_iter = 20
for iter in xrange(num_iter) :
	for line_pair in bitext :
		f_len = len(line_pair[0])
		e_len = len(line_pair[1])
		lm_pair = str(f_len) + "," + str(e_len)

		# f sentence
		for i in xrange(f_len) :
			f_word = line_pair[0][i]

			# iterate all j
			sum_j = 0.0
			for each_j in xrange(e_len) :
				sum_j += math.exp(l * h[lm_pair][str(i)][str(each_j)])

			# e sentence

			for j in xrange(e_len) :
				# calculate delta
				e_word = line_pair[1][j]
				delta = q[lm_pair][str(i)][str(j)] * t[f_word][e_word] / sum_j

				# add counters
				c_ef_pair[f_word][e_word] += delta
				c_e_all[e_word] += delta
				#c_len_pair[lm_pair][str(i)][str(j)] += delta
				#c_len_all[lm_pair] += delta
				h[lm_pair][str(i)][str(j)] = - math.fabs(float(i) / f_len - float(j) / e_len)
				b[lm_pair][str(i)][str(j)] = math.exp(l * h[lm_pair][str(i)][str(j)]) / sum_j
				if j == 0 :
					q[lm_pair][str(i)][str(j)] = p0
				else :
					q[lm_pair][str(i)][str(j)] = (1 - p0) * b[lm_pair][str(i)][str(j)]



	# update params
	# for lm_pair in q.keys() :
	# 	for i, i_iter in enumerate(q[lm_pair].keys()) :
	# 		for j_iter in q[lm_pair][i_iter] :
	# 			q[lm_pair][i_iter][j_iter] = 
				#c_len_pair[lm_pair][i_iter][j_iter] = 0.0
		#c_len_all[lm_pair] = 0.0

	for f_word in t.keys() :
		for e_word in t[f_word].keys() :
			t[f_word][e_word] = float(c_ef_pair[f_word][e_word]) / c_e_all[e_word]
			c_ef_pair[f_word][e_word] = 0.0
	# Clear counters after each iteration
	for e_word in c_e_all.keys() : 
		c_e_all[e_word] = 0.0


# use params to calculate alignment
for line_pair in bitext :
	f_len = len(line_pair[0])
	e_len = len(line_pair[1])
	# for each length pair, maintain a 2 dim array
	lm_pair = str(f_len) + "," + str(e_len)

	for i in xrange(f_len) :
		f_word = line_pair[0][i]
		best_a_pos = -1
		best_a_pr = 0.0
		for j in xrange(e_len) :
			e_word = line_pair[1][j]
			pr = q[lm_pair][str(i)][str(j)] * t[f_word][e_word]
			if pr > best_a_pr :
				best_a_pr = pr
				best_a_pos = j

		if best_a_pos == 0:
			continue;
		print str(i) + "-" + str(best_a_pos - 1),;

	print

