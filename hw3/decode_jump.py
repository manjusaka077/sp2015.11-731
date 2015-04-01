#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple
import copy

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=1, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
opts = parser.parse_args()

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
hypothesis = namedtuple('hypothesis', 'logprob, lm_state, predecessor, phrase, translated')


def calc_prob(f, stacks) :
    # for i, stack in enumerate(stacks[:-1]):
    # extend the top s hypotheses in the current stack
    for length, stack in enumerate(stacks[:-1]):
        # print stack
        # extend the top s hypotheses in the current stack
        # for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h: h.logprob): # prune
        # for h in heapq.nlargest(opts.s, stack, key=lambda h: h.logprob): # prune
        for key in stack.keys() :
            h = stack[key]
        # for h in stack :
            # print "==========================================="
            # print h.lm_state
            for i in xrange(len(f)) :
                if i - 1 > length :
                    break
                if h.translated[i] != 1 :

                    for j in xrange(i,len(f)):

                        if h.translated[j] == 1 :
                            break
                        translated = copy.deepcopy(h.translated)

                        if f[i : j+1] in tm:

                            for k in xrange(i, j+1) :
                                translated[k] = 1
                            # print translated
                            # if h.translated[3] == 1 and i == 7 :
                            #     print translated

                            # print "f :" + str(f[i : j+1])
                            for phrase in tm[f[i : j+1]]:

                                # if h.translated[3] == 1 and i == 7 :
                                #     print phrase

                                logprob = h.logprob + phrase.logprob
                                lm_state = h.lm_state

                                # if h.translated[3] == 1 and i == 7 :
                                #     print "old state : " + str(lm_state)

                                for word in phrase.english.split():
                                    (lm_state, word_logprob) = lm.score(lm_state, word)

                                    # if h.translated[3] == 1 and i == 7 :
                                    #     print "new state : " + str(lm_state)
                                    logprob += word_logprob

                                trans_len = sum(translated)
                                # if h.translated[3] == 1 and i == 7 :
                                #     print trans_len

                                logprob += lm.end(lm_state) if trans_len == len(f) else 0.0
                                # print translated
                                
                                new_hypothesis = hypothesis(logprob, lm_state, h, phrase, translated)
                                # stacks[trans_len].append(new_hypothesis)
                                if lm_state not in stacks[trans_len] or stacks[trans_len][lm_state].logprob < logprob: # second case is recombination
                                    stacks[trans_len][lm_state] = new_hypothesis 

                    # break
    # print length
    # for key in stacks[-7] :
    #     print key
    #     print stacks[-7][key].translated
        # print stacks[-3][key].predecessor

def decode(opts) :

    sys.stderr.write('Decoding %s...\n' % (opts.input,))
    input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]



    for f in input_sents:
        # The following code implements a DP monotone decoding
        # algorithm (one that doesn't permute the target phrases).
        # Hence all hypotheses in stacks[i] represent translations of 
        # the first i words of the input sentence.
        # HINT: Generalize this so that stacks[i] contains translations
        # of any i words (remember to keep track of which words those
        # are, and to estimate future costs)
        initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, [0 for i in f])

        stacks = [{} for _ in f] + [{}]
        # stacks = [[] for _ in f] + [[]]
        # stacks[0] = [initial_hypothesis]
        stacks[0][lm.begin()] = initial_hypothesis

        # paths = [[word] for word in f]
        # add_word(f, paths, stacks)
        # print paths
 
        calc_prob(f, stacks)

        # find best translation by looking at the best scoring hypothesis
        # on the last stack
        winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

        def extract_english_recursive(h):
            return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)
        print extract_english_recursive(winner)

        if opts.verbose:
            def extract_tm_logprob(h):
                return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
            tm_logprob = extract_tm_logprob(winner)
            sys.stderr.write('LM = %f, TM = %f, Total = %f\n' % 
                (winner.logprob - tm_logprob, tm_logprob, winner.logprob))


def main() :
    decode(opts)



if __name__ == '__main__':
    main()
