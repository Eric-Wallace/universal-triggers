# Attacking SQuAD Models

## Files and Usage
This folder contains the following files: 
+ `squad.py` Creates the triggers using the optimization procedure described in the paper.  
+ `universal_adversarial_sentences.txt` contains the adversarial sentences used in the paper. You can copy and paste these into the variable `adv_token_idxs` in `squad.py` to test out different attacks.


## Models

We have pre-trained models available for use (see `squad.py` for how to use a model archive):
+ `bidaf` as described in the paper using GloVe word embeddings. The model archive is available here https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-glove-2019.05.09.tar.gz