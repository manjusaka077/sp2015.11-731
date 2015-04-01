#My method
In this homework, I tried three methods to switch phrases during translation.
- In `decode_jump.py` I allow the decoder to jump forward to translate some word ahead. `stacks[i] `stores result with length of `i`. When I set the longest jump step to be 2, I got a logprob of -5297. But this method has one flaw: the decoder may keep jumping and translate towards the end of the sentence before go back to translate first several words in the sentence. And if some word only occurs as a part of a phrase in the translation model, the decoder may jump over this word and when it comes back, there is no corresponding translation for this single word. And the decoding process will fail. (When setting jump step to 3, this flaw appears.)

- In `decode_adjacent.py` I decide to only allow the decoder jump to its next phrase. And if it translate the next phrase first, next time it will not keep jump but have to finish translating the word it ignores first. I thought it might be what the homeword instruction talks about by saying "capable of swapping adjacent phrase pairs". Unfortunately, this method only get a logprob of less than -5400 :(

- In `decode_switch.py` I decide to switch the phrase after translation, say, change the order of the generated english sentence. Thanks Hao for providing this idea:) I borrow the lm score grading part from the grade script and let it be the benchmark for inserting new generated english phrase to existed sentence. This method works surprisingly good than switching before translation.


#Original
There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model
