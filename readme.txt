This is the specification file for conditional seq2seq VAE training and validation dataset - English tense

1. train.txt
The file is for training. There are 1227 training pairs.
Each training pair includes 4 words: simple present(sp), third person(tp), present progressive(pg), simple past(p).


2. test.txt
The file is for validating. There are 10 validating pairs.
Each training pair includes 2 words with different combination of tenses.
You have to follow those tenses to test your model.

Here are to details of the file:

sp 0-> p 3
sp 0-> pg 2
sp 0-> tp 1
sp 0-> tp 1
p  3-> tp 1
sp 0-> pg 2
p  3-> sp 0
pg 2-> sp 0
pg 2-> p 3
pg 2-> tp 1

