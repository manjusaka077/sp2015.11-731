#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import copy

from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer() # stem

from nltk.corpus import wordnet as wn # synonymy

from nltk.tokenize import WordPunctTokenizer # tokenizer

alpha = 0.6 # precision and recall
beta = 3 # importance of fragmentation penalty
gamma = 0.3 # fragmentation weight
syn = 1 # synonymy penalty

# Match word return alignment
def word_matches(h, ref) :
    match_score = 0.0
    match_pair = []
    left_word_index = [i for i, word in enumerate(ref)]

    for i, word in enumerate(h) :
        if len(left_word_index) == 0 :
            break     
        for index in left_word_index :
            ref_word = ref[index]
            # Exact match
            if word == ref_word: 
                match_pair.append([i, index])
                left_word_index.remove(index)
                match_score += 1
                break
            # Synonymy match
            if word in get_lemmas(ref_word) :
                match_pair.append([i, index])
                left_word_index.remove(index)
                match_score += 1 * syn
                break
            # Stem match
            # if st.stem(word) is st.stem(eachword) :
            #     #print "stem : " + word + "\t" + st.stem(word) + "\t" + eachword
            #     match.append(word)
            #     ref_copy.remove(eachword)
            #     break
    return match_pair, match_score

# Match word only return number
def num_word_matches(h, ref) :

    match = 0
    ref_copy = copy.deepcopy(ref)

    for word in h :
        if len(ref_copy) == 0 :
            break
        # Exact match
        if word in ref_copy :
            match += 1
            ref_copy.remove(word)
            continue
        # Synonymy match
        for eachword in ref_copy :
            if word in get_lemmas(eachword):
                match += 1 * syn
                ref_copy.remove(eachword)
                break
    return match

# Lemmas of all meanings of this word
# rset is a set of words
def get_lemmas(word) :
    
    lemmas = set()
    for syn in wn.synsets(word) :
        lemmas.update(set([name.lower() for name in syn.lemma_names]))
    return lemmas


# Calculate fragment
# Natch_pair is a list contains pair of alignments from h to ref
def frag(match_pair) : 

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

# Count ngram
# Sentence is a list of words
def ngram(sentence, n) :
    ngram = []
    for i in xrange(len(sentence) - n + 1) :
        ngram.append(" ".join(sentence[i : i + n]))
    return set(ngram)
    #return ngram

def ngram_matches(h, ref, n) :
    ngram_match = 0
    h_ngram = ngram(h, n)
    ref_ngram = ngram(ref, n)
    # for pair in h_ngram :
    #     if pair in ref_ngram :
    #         ngram_match += 1
    ngram_match = h_ngram & ref_ngram
    return len(ngram_match), len(h_ngram)

# Calculate Fmean
def fscore(h, ref):
    """
    h: list of words in h
    ref: list of words in ref
    """

    h_len = len(h)
    ref_len = len(ref)

    # Unigram match
    h_match, h_num_match = word_matches(h, ref)
    ref_num_match = num_word_matches(ref, h)

    # ngram match
    h_bigram_score, h_bigram_num = ngram_matches(h, ref, 2)
    ref_bigram_score, ref_bigram_num = ngram_matches(ref, h, 2)
    #h_trigram_score, h_trigram_num = ngram_matches(h, ref, 3)
    #ref_trigram_score, ref_trigram_num = ngram_matches(ref, h, 3)
    #print "Exact match : %i / %i = %f" %(h_num_match, h_len, h_num_match / float(h_len))
    #print "Bigram match : %i / %i = %f" %(h_bigram_score, h_bigram_num, h_bigram_score / float(h_bigram_num))
    #print "Trigram match : %i / %i = %f" %(h_trigram_score, h_trigram_num, h_trigram_score / float(h_trigram_num))

    # Precision and recall
    #p = (float(h_num_match) ) / (h_len )
    #r = (float(ref_num_match) ) / (ref_len )
    p = (float(h_num_match) + h_bigram_score) / (h_len + h_bigram_num )
    r = (float(ref_num_match) + ref_bigram_score) / (ref_len + ref_bigram_num)
    #p = (float(h_num_match) + h_bigram_score + h_trigram_score) / (h_len + h_bigram_num + h_trigram_num)
    #r = (float(ref_num_match) + ref_bigram_score + ref_trigram_score) / (ref_len + ref_bigram_num + ref_trigram_num)

    #print "Score : %f" %p
    if not (p == 0.0 or r == 0.0) :
        f_mean = alpha * p * r / (alpha * p + (1 - alpha) * r)
    else :
        f_mean = 0.0
    #print "Fmean : %f" %f_mean
    # Fragmentation
    if h_num_match > 1 :
        fragmentation = (frag(h_match) - 1.0) / (h_num_match - 1.0)
        DF = gamma * pow(fragmentation, beta)
        final = f_mean * (1 - DF)
    else :
        final = f_mean

    #print "Final : %f" %final 
    return final


 
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
        h1_score = fscore(h1, ref)
        h2_score = fscore(h2, ref)

        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
