# My method
## Features
I use the provided 4 log probs, previous word, and next word, as recommaned for base line code.

## Matrix
For the sparse features, I first create lil_matrix, read file and add value to matrix, then converted to a csr_matrix.

For the weight vector, I use numpy array.

## Parameter
I use 3 tunable paramters: alpha for learning rate, gamma for margin, and number of iterations. Because of the time taken to run the code each time, I do not have the chance to tune these parameters. Hope this can be done during the extension 2 weeks.

## Speed
It must be the part I spend most time in this homework! At first my code as so slow that it takes almost 1 second to train each sample which is totally a disaster. Then I use profiler to look into the code and fix several problems (like the matrix cannot be saved as sparse matrix and the choice of matrix type). But now I still suffer from the high cost of filling in the sparse matrix. Hope I can figure out a method to avoid chaning sparsity of the feature matrix during extension days :)

The current version takes about 1 hour to train the whole dataset for 1 iteration.

# Updates
## Achieve Baseline(05/06/2015)
In `rerank_dict.py` I abandon matrices and use dictionary to store weights and feature vectors. It significantly improves the speed. Now doing 2 iterations uses about 5 minutes. After trying out some values for the alpha and gamma, I choose 0.001 for alpha and 0.1 for gamma.

# Original
There are three Python programs here (`-h` for usage):

 - `./rerank` a simple reranker that simply sorts candidate translations on log p(czech|english)
 - `./grade` computes the mean reciprocal rank of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./rerank | ./check | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/train.input` is the input side of training set in the format described on the homework webpage

 - `data/train.refs` are the references to the training set, giving the correct czech translation for the highlighted phrase in each sentence

 - `data/train.parses` are dependency parses of the training sentences, provided for convenience. (Note: these files are provided in gzip format to avoid the space limitations imposed by github)

 - `data/dev+test.input` is the input side of both the dev and test sets

 - `data/dev.refs` are the references to the dev set, which is the first half of the above dev+test file

 - `data/dev+test.parses` are dependency parses of the dev and test sentences, provided for convenience

 - `data/ttable` is the phrase translation table which contains candidates that you will rerank

 If you want the raw parallel data used to build the training data and translation tables English-Czech data (for example, to build word vectors), it is available at http://demo.clab.cs.cmu.edu/sp2015-11731/parallel.encs .
