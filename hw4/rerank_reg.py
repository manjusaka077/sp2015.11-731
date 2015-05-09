#!/usr/bin/env python
import sys
import argparse
from collections import defaultdict
from utils import read_ttable

import os
import numpy as np
import cPickle as cp
import operator
import copy



# Parse inpur parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='data/train.input')
parser.add_argument('--refer', '-r', default='data/train.refs')
parser.add_argument('--test', '-d', default='data/dev+test.input')
parser.add_argument('--num_sents', '-n', default='0')
parser.add_argument('--ttable', '-t', default='data/ttable')
parser.add_argument('--save', '-s', default='data/weights') # save weights
parser.add_argument('--load', '-l', default='') # load weights from file
parser.add_argument('--alpha', '-a', default='0.01')
parser.add_argument('--gamma', '-g', default='0.1')
parser.add_argument('--mu', '-u', default='0.0001')
parser.add_argument('--iteration', '-x', default='1')
args = parser.parse_args()

# Global parameter
alpha = float(args.alpha)
gamma = float(args.gamma)
mu = float(args.mu)
init_val = 0.001

# Global data structures
num_sents = int(args.num_sents)
if num_sents == 0 :
	num_sents = sys.maxint
translation_table = read_ttable(args.ttable)
with open(args.refer, 'r') as f_refer :
	train_ref = [line.decode('utf-8').rstrip() for line in f_refer]
with open(args.input, 'r') as f_input :
	train_data = [line.decode('utf-8').rstrip() for line in f_input]
print >>sys.stderr, "Finish reading file reference and input file."

# @profile
def dot(diff, weights) :
	s = 0.0
	for k in diff:
		s += diff[k] * weights[k]
	return s

# @profile
def minus(f_best, f_other) :
	s = {}
	for k in f_other:
		s[k] = f_best[k] - f_other[k]
	return s

def dot_number(w, num) :
	for k in w.keys() :
		w[k] *= num
	return w


# Funcs for features
def add_feature(feature_str, weights) :
	weights[feature_str] = init_val
	return weights

def create_feature(left_context, right_context, phrase, target) :
	if len(left_context) > 0 :
		prev_word = left_context.split(" ")[-1]
	else :
		prev_word = "None"
	if len(right_context) > 0 :
		next_word = right_context.split(" ")[0]
	else :
		next_word = "None"
	return "src:" + phrase + "_tgt:" + target + "_prev:" + prev_word, \
		"src:" + phrase + "_tgt:" + target + "_next:" + next_word

def read_features() :
	weights = {'log_lex_prob_tgs': init_val, 'log_prob_sgt': init_val, \
					'log_lex_prob_sgt': init_val, 'log_prob_tgs': init_val}
	for line in train_data :
		left_context, phrase, right_context = [part.strip() for part in line.split('|||')]
		targets = translation_table[phrase].keys()
		for each_y in targets :
			new_prev_feature, new_next_feature = \
				create_feature(left_context, right_context, phrase, each_y)
			weights = add_feature(new_prev_feature, weights)
			weights = add_feature(new_next_feature, weights)
	print >>sys.stderr, "Created weights."
	return weights

# For SGD
def calc_loss(diff, weights) :
	loss = 0.0
	# print >>sys.stderr, f_probs_best
	# print >>sys.stderr, f_probs_others
	# print >>sys.stderr, diff
	tmp = dot(diff, weights)
	# print >>sys.stderr, tmp
	return max(0, gamma - tmp)
		
# Train
# @profile
def train(weights) :
	print >>sys.stderr, "Begin training using data in %s..." %args.input

	rand_permut = np.random.permutation(min(len(train_ref), num_sents))

	A = dict((key, 0) for key in weights)
	count = 0

	for num, i in enumerate(rand_permut) :

		line = train_data[i]
		left_context, phrase, right_context = [part.strip() for part in line.split('|||')]

		y_star = train_ref[i]
		y_star_prev_feature, y_star_next_feature = \
			create_feature(left_context, right_context, phrase, y_star)

		f_probs_best = copy.deepcopy(translation_table[phrase][y_star])
		f_probs_best[y_star_prev_feature] = 1
		f_probs_best[y_star_next_feature] = 1

		y_others = copy.deepcopy(translation_table[phrase].keys())
		y_others.remove(y_star)

		# print >>sys.stderr, y_star

		for j, each_y in enumerate(y_others) :
			# print >>sys.stderr, each_y

			each_y_prev_feature, each_y_next_feature = \
				create_feature(left_context, right_context, phrase, each_y)
			f_probs_other = copy.deepcopy(translation_table[phrase][each_y])

			f_probs_other[y_star_prev_feature] = 0
			f_probs_other[y_star_next_feature] = 0

			f_probs_other[each_y_prev_feature] = 1
			f_probs_other[each_y_next_feature] = 1	

			f_probs_best[each_y_prev_feature] = 0
			f_probs_best[each_y_next_feature] = 0	

			diff = minus(f_probs_best, f_probs_other)
			loss = calc_loss(diff, weights)
			
			count += 1
			if loss != 0 :
				# print >>sys.stderr, "Update"
				# update_val = minus(f_probs_other, f_probs_best)
				diff = dot_number(diff, alpha)
				for k in diff :
					# L2 regularization
					weights[k] *= pow(1 - alpha * 2 * mu, count - A[k])
					weights[k] += diff[k]
					A[k] = count

		sys.stderr.write('%d\r' %(num + 1))

	# Update regularization for all weights
	for k in weights :
		weights[k] *= pow(1 - alpha * 2 * mu, count - A[k])

	print >>sys.stderr
	# if os.path.isfile(args.save) :
	cp.dump(weights, open(args.save, 'w'))
	print >>sys.stderr, "Weights saved to %s." %args.save

	return weights


# Predict
# @profile
def predict(weights) :
	print >>sys.stderr, "Begin testing using samples in %s..." %args.test

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

				f_probs = copy.deepcopy(translation_table[phrase][each_y])
				f_probs[new_prev_feature] = 1
				f_probs[new_next_feature] = 1

				score = 0.0
				for key in f_probs :
					if key in weights :
						score += f_probs[key] * weights[key]
				y_score[each_y] = score

			candidates = [target for target, features in \
				sorted(y_score.iteritems(), key=operator.itemgetter(1), reverse=True)]
			print ' ||| '.join(candidates).encode('utf-8')

# @profile
def main() :
	if (os.path.isfile(args.load)) :
		weights = cp.load(open(args.load, 'r'))
		print >>sys.stderr, "Load weights with dimension of %i" %len(weights)
	else :
		weights = read_features()
	
	for it in xrange(int(args.iteration)) :
		weights = train(weights)
		# for key in weights :
		# 	if weights[key] != 0 :
		# 		print >>sys.stderr, weights[key]

	predict(weights)


if __name__ == '__main__':
	main()