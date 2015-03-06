#My Solution
Data processing:

 - I use the NLTK tokenizer to tokenize the input text.

Model:

 - `meteor.py` shows a simple METEOR method, which calculate precision, recall and calculate Fmean. The only free parameter is alpha which controls the trade off between precision and recall. 
 - Then I add Wordnet dataset for synonymy match. For each word in reference sentence, I use all the lemmas for all its meaning as its synonymy set. Then if any word in hypothesis sentence is in this synonymy set, I will consider it as a match.
 - `meteor_full.py` add the fragmentation part to the simple METEOR method. When looking for match (exact or synonymy match) words, I record the alignment between word in hypothesis and reference. With this record I seperate matched words into fragments.
 - `meteor_ngram.py` add bigram for the matching part. I use both the number of unigram match (include exact match and synonymy match) and the number of bigram match togetherto calculate precision.

Some other failed attempts:

 - I try to use the porter stemmer provided by NLTK and add stem match, but the accuracy decrease, which is pretty strange...
 - I try to give penalty to synonymy match, it works well on the simple METEOR model but fails on the model with fragmentation. The reason might be that the fragmentation already give penalty to the score.
 - Add trigram additional to the bigram version model also lowers the accurcacy. Might be the same reason with the synonymy penalty: the score becomes too much lower.
 - I found the predicion for 0 label is pretty unaccurate. So I try to broad the range of "the two translation result are the same" by let the model print 1 only if h1_score < h2_score - threshold and vice versa. But it fails.

 
#Original readme

There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
