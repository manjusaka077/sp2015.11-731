#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import copy

from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer() # stem

from nltk.corpus import wordnet as wn # synonymy

from nltk.tokenize import WordPunctTokenizer # tokenizer

alpha = 0.6
beta = 3
gamma = 0.3
syn = 1

# Match word
def word_matches(h, ref) :
    match_score = 0.0
    match_pair = []
    left_word_index = [i for i, word in enumerate(ref)]

    for i, word in enumerate(h) :

        if len(left_word_index) == 0 :
            break
       
        for index in left_word_index :
            ref_word = ref[index]
            # Exact match or syn match
            if word == ref_word: 
                match_pair.append([i, index])
                left_word_index.remove(index)
                match_score += 1
                break
            # stem match
            # stem_word = st.stem(word)
            # if stem_word == st.stem(eachword) :
            #     #print "stem : " + word + "\t" + st.stem(word) + "\t" + eachword
            #     match_pair.append([i, index])
            #     left_word_index.remove(index)
            #     match_score += 1
            #     break
            if word in get_lemmas(ref_word) :
                match_pair.append([i, index])
                left_word_index.remove(index)
                match_score += 1 * syn
                break

    return match_pair, match_score

# Match word
def num_word_matches(h, ref) :

    match = 0

    ref_copy = copy.deepcopy(ref)
   
    for word in h :
        if len(ref_copy) == 0 :
            break
        # exact match
        if word in ref_copy :
            match += 1
            ref_copy.remove(word)
            continue
        # stem match
        # stem_word = st.stem(word)
        for eachword in ref_copy :
        #     if stem_word == st.stem(eachword) :
        #         match += 1
        #         ref_copy.remove(eachword)
        #         break
        # synonymy match
            if word in get_lemmas(eachword):
                match += 1
                ref_copy.remove(eachword)
                break
    return match

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

    h_match, h_num_match = word_matches(h, ref)
    ref_num_match = num_word_matches(ref, h)


    p = float(h_num_match) / h_len
    r = float(ref_num_match) / ref_len
    #print str(p) + "\t" + str(r)
    if not (p == 0.0 or r == 0.0) :
        f_mean = alpha * p * r / (alpha * p + (1 - alpha) * r)
    else :
        f_mean = 0.0

    # fragmentation
    if h_num_match > 1 :
        fragmentation = (frag(h_match) - 1) / float(h_num_match - 1)
        DF = gamma * pow(fragmentation, beta)
        final = f_mean * (1 - DF)
    else :
        final = f_mean

    return final

# Calculate fragment
def frag(match_pair) :
    """
    match_pair: pair of alignments from h to ref
    """
    h = match_pair[0][0]
    ref = match_pair[0][1]
    del match_pair[0]

    for i in xrange(1, len(match_pair) + 1) :
        if [h + i, ref + i] in match_pair :
            match_pair.remove([h + i, ref + i])
        else :
            break

    fragment = 1

    if len(match_pair) != 0 :
        fragment += frag(match_pair)

    return fragment

 
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
