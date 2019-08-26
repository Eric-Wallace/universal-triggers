## Files and Usage
This folder contains the following files: 
+ `snli.py` used to generate attacks against SNLI models. 

## Models

We have pre-trained models available for use:
+ `Decomposable Attention` as described in our paper (Section 3) using GloVe word embeddings. The AllenNLP model archive is available here https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz, see `snli.py` for how to use this model archive. 
+ `Decomposable Attention with ELMo`. The model archive is available https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz
+ `ESIM Model with Glove` as described in the paper using GloVe word embeddings https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz.