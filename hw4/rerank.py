#!/usr/bin/env python
import sys
import argparse
from collections import defaultdict
from utils import read_ttable

from copy import deepcopy
import random
import operator
import cPickle as cp
import os

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix

# Parse inpur parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='data/train.input')
parser.add_argument('--refer', '-r', default='data/train.refs')
parser.add_argument('--test', '-d', default='data/dev+test.input')
parser.add_argument('--num_sents', '-n', default='0')
parser.add_argument('--ttable', '-t', default='data/ttable')
parser.add_argument('--save', '-s', default='data/weights') # save weights
parser.add_argument('--load', '-l', default='') # load weights from file
parser.add_argument('--gamma', '-g', default='0.1')
parser.add_argument('--iteration', '-x', default='1')
args = parser.parse_args()

# Global parameter
alpha = 0.01
gamma = float(args.gamma)

# Global data structures
num_sents = int(args.num_sents)
if num_sents == 0 :
	num_sents = sys.maxint
translation_table = read_ttable(args.ttable)
with open(args.refer, 'r') as f_refer :
	train_ref = [line.decode('utf-8').rstrip() for line in f_refer]
with open(args.input, 'r') as f_input :
	train_data = [line.decode('utf-8').strip() for line in f_input]
print >>sys.stderr, "Finish reading file reference and input file."

# Funcs for SGD
# @profile
def calc_loss(f_y_best, f_y_others, w) :
	diff = f_y_best - f_y_others
	tmp = diff.dot(w.T)
	loss = 0.0
	for row in xrange(tmp.shape[0]) :
		loss += max(0, gamma - tmp[row, 0])
	return loss

# Funcs for features
def add_feature(feature_str, feature_map) :
	if not feature_str in feature_map :
		feature_map[feature_str] = len(feature_map)

def create_feature(left_context, right_context, phrase, target) :
	if len(left_context) > 0 :
		prev_word = left_context[-1]
	else :
		prev_word = ""
	if len(right_context) > 0 :
		next_word = right_context[0]
	else :
		next_word = ""
	return phrase + "_" + target + "_prev:" + prev_word, \
		phrase + "_" + target + "_next:" + next_word

def read_features() :
	feature_map = {'log_lex_prob_tgs': 0, 'log_prob_sgt': 1, \
					'log_lex_prob_sgt': 2, 'log_prob_tgs': 3}
	for line in train_data :
		left_context, phrase, right_context = [part.strip() for part in line.split('|||')]
		targets = translation_table[phrase].keys()
		for each_y in targets :
			new_prev_feature, new_next_feature = \
				create_feature(left_context, right_context, phrase, each_y)
			add_feature(new_prev_feature, feature_map)
			add_feature(new_next_feature, feature_map)
	print >>sys.stderr, "Created feature map."
	return feature_map

# Train
# @profile
def train(feature_map, weights) :
	print >>sys.stderr, "Begin training using data in %s..." %args.input

	rand_permut = np.random.permutation(min(len(train_ref), num_sents))

	for num, i in enumerate(rand_permut) :
		line = train_data[i]
		left_context, phrase, right_context = [part.strip() for part in line.split('|||')]

		y_star = train_ref[i]
		y_star_prev_feature, y_star_next_feature = \
			create_feature(left_context, right_context, phrase, y_star)

		num_rows = len(translation_table[phrase])
		# f_array = np.zeros((num_rows, len(feature_map)))
		# f_array[:, 0:4] = np.tile(np.array( \
		# 	translation_table[phrase][y_star].values()), (num_rows, 1))
		# f_array[:, feature_map[y_star_prev_feature]] = 1
		# f_array[:, feature_map[y_star_next_feature]] = 1
		# f_probs_best = lil_matrix(f_array)

		f_probs_best = lil_matrix((num_rows, len(feature_map)))
		f_probs_best[:, 0:4] = np.tile(np.array( \
			translation_table[phrase][y_star].values()), (num_rows, 1))
		f_probs_best[:, feature_map[y_star_prev_feature]] = 1
		f_probs_best[:, feature_map[y_star_next_feature]] = 1

		f_probs_best = f_probs_best.tocsr()
		# f_probs_best = lil_matrix((1, len(feature_map)))
		# f_probs_best[0, 0:4] = np.array( \
		# 	translation_table[phrase][y_star].values())
		# f_probs_best[0, feature_map[y_star_prev_feature]] = 1
		# f_probs_best[0, feature_map[y_star_next_feature]] = 1
		# update_val = lil_matrix(np.zeros((1, len(feature_map))))
		# loss = 0.0

		
		f_probs_others = lil_matrix((num_rows, len(feature_map)))
		# f_array_other = np.zeros((num_rows, len(feature_map)))

		y_others = translation_table[phrase].keys()
		for j, each_y in enumerate(y_others) :
			each_y_prev_feature, each_y_next_feature = \
				create_feature(left_context, right_context, phrase, each_y)

			# f_probs_others = lil_matrix((1, len(feature_map)))
			# f_probs_others[0, 0:4] =  np.array( \
			# 	translation_table[phrase][each_y].values())
			# f_probs_best[0, feature_map[each_y_prev_feature]] = 1
			# f_probs_best[0, feature_map[each_y_next_feature]] = 1
			# diff = f_probs_others - f_probs_best
			# update_val += diff
			# loss += calc_loss(diff, weights)

			# f_probs_others[j, 0:4] =  np.array( \
			# 	translation_table[phrase][each_y].values())
			f_probs_others[j, 0:4] = translation_table[phrase][each_y].values()		
			f_probs_others[j, feature_map[each_y_prev_feature]] = 1
			f_probs_others[j, feature_map[each_y_next_feature]] = 1

			# f_array_other[j, 0:4] = np.array( \
			# 	translation_table[phrase][each_y].values())
			# f_array_other[j, feature_map[each_y_prev_feature]] = 1
			# f_array_other[j, feature_map[each_y_next_feature]] = 1
		f_probs_others = f_probs_others.tocsr()
		# f_probs_others = lil_matrix(f_array_other)

		diff = f_probs_others - f_probs_best 
		update_val = csr_matrix(diff.sum(0))

		if calc_loss(f_probs_best, f_probs_others, weights) != 0 :
			weights -= update_val * alpha

		sys.stderr.write('%d\r' %(num + 1))

	print >>sys.stderr
	# if os.path.isfile(args.save) :
	cp.dump(weights, open(args.save, 'w'))
	print >>sys.stderr, "Weights saved to %s." %args.save
	
# Predict
# @profile
def predict(feature_map, weights) :

	print >>sys.stderr, "Begin testing using samples in %s..." %args.test

	# with open(args.test, 'r') as f_test :
	# 	test_data = [line.decode('utf-8').strip() for line in f_test]
	wt = weights.T
	with open(args.test, 'r') as f_test :
		for i, line in enumerate(f_test) :
		# for line in test_data :
			left_context, phrase, right_context = [part.strip() \
				for part in line.decode('utf-8').strip().split('|||')]
			targets = translation_table[phrase].keys()
			y_score = {}
			for each_y in targets :
				new_prev_feature, new_next_feature = \
					create_feature(left_context, right_context, phrase, each_y)
				f_probs = lil_matrix((1, len(feature_map)))	
				f_probs[0, 0:4] = np.array( \
					translation_table[phrase][each_y].values())
				if new_prev_feature in feature_map :
					f_probs[0, feature_map[new_prev_feature]] = 1
				if new_next_feature in feature_map :
					f_probs[0, feature_map[new_next_feature]] = 1 
				# print >>sys.stderr, f_probs.shape
				y_score[each_y] = f_probs * wt
			candidates = [target for target, features in \
				sorted(y_score.iteritems(), key=operator.itemgetter(1), reverse=True)]
			print ' ||| '.join(candidates).encode('utf-8')


# @profile
def main() :
	if (os.path.isfile(args.load)) :
		weights = cp.load(open(args.load, 'r'))
		print >>sys.stderr, "Load weights with dimension of %s" %str(weights.shape)
		feature_map = cp.load(open(args.load + "_features", 'r'))
	else :
		feature_map = read_features()
		weights = np.random.rand(1, len(feature_map)) / 10
		cp.dump(feature_map, open(args.save + "_features", 'w'))
		cp.dump(weights, open("data/weights_init", 'w'))
	
		for it in xrange(int(args.iteration)) :
			train(feature_map, weights)
	
	predict(feature_map, weights)


if __name__ == '__main__':
	main()