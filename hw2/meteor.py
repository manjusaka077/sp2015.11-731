#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import copy

#from nltk.stem.lancaster import LancasterStemmer
#st = LancasterStemmer() # stem
from nltk.stem.porter import *
st = PorterStemmer()

from nltk.corpus import wordnet as wn # synonymy

from nltk.tokenize import WordPunctTokenizer # tokenizer

alpha = 0.6

# Match word
def num_word_matches(h, ref) :

    match = 0
    stem = 0
    syns = 0

    ref_copy = copy.deepcopy(ref)
    h_copy = copy.deepcopy(h)

    
    for word in h :
        if len(ref_copy) == 0 :
            break
        # exact match
        if word in ref_copy :
            match += 1
            ref_copy.remove(word)
            #h_copy.remove(word)
            continue
        # stem match
        stem_word = st.stem(word)
        for eachword in ref_copy :
            if stem_word == st.stem(eachword) :
                stem += 1
                ref_copy.remove(eachword)
                break
        # synonymy match
            if word in get_lemmas(eachword):
                syns += 1
                ref_copy.remove(eachword)
                break
    #print "match : %i stem : %i syns: %i" %(match, stem, syns)
    return match + stem + syns


def get_lemmas(word) :
    # rset is a set of words
    lemmas = set()
    for syn in wn.synsets(word) :
        lemmas.update(set([name.lower() for name in syn.lemma_names]))
    return lemmas

# Calculate Fmean
def fmean(h, ref):
    """
    h: list of words in h
    ref: list of words in ref
    """

    h_len = len(h)
    ref_len = len(ref)

    h_num_match = num_word_matches(h, ref)
    ref_num_match = num_word_matches(ref, h)

    p = float(h_num_match) / h_len
    r = float(ref_num_match) / ref_len
    #print str(p) + "\t" + str(r)
    if not (p == 0.0 or r == 0.0) :
        f_mean = alpha * p * r / (alpha * p + (1 - alpha) * r)
    else :
        f_mean = 0.0
    #print f_mean
    return f_mean

# Calculate fragment
def frag(h_match, ref_match) :
    """
    h_match : list of matched words in h
    ref_match : list of matched words in ref
    """
 
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
                #yield [sentence.strip().lower().split() for sentence in pair.split(' ||| ')]
                yield [WordPunctTokenizer().tokenize(sentence.strip().lower()) for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1_score = fmean(h1, ref)
        h2_score = fmean(h2, ref)


        print(-1 if h1_score > h2_score else # \begin{cases}
                (1 if h1_score < h2_score
                    else 0)) # \end{cases}

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
