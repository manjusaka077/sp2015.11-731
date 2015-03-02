#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
 

alpha = 0.9

# DRY
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!

# Calculate Fmean
def fmean(h, ref):
    '''
    h: type: list, hypothesis
    ref: type: list, reference
    '''

    h_len = len(h)
    ref_len = len(ref)
    rset = set(ref)
    hset = set(h)
    h_match = word_matches(h, rset)
    ref_match = word_matches(ref, hset)
    p = float(h_match) / h_len
    r = float(ref_match) / ref_len
    #print str(p) + "\t" + str(r)
    if not (p == 0.0 and r == 0.0) :
        f_mean = alpha * p * r / (alpha * p + (1 - alpha) * r)
    else :
        f_mean = 0.0
    return f_mean



 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1_score = fmean(h1, ref)
        h2_score = fmean(h2, ref)


        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
