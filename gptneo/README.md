# Attacking GPT-2

## Files and Usage

The GPT-2 code is not in AllenNLP and consequently is a bit separate from the rest of the repo. It is a good example if you want to create triggers for your task/model/codebase.

This folder contains the following files: 
+ `create_adv_token.py` Creates the triggers using the optimization procedure described in the paper.  
+ `sample_from_gpt2.py` creates output samples from GPT-2, optionally using the trigger.

You can easily swap between small and big GPT-2 models by changing the argument to `GPT2LMHeadModel.from_pretrained()`.


## Future Work

Some future TODOs, feel free to try it out:
+ Try different concepts besides racism, e.g., get GPT-2 to generate fake Tesla stock reports, fake news, articles, sports, technology, hate speech, etc. You can do this by changing the target_texts to have the content you want. This may be better than fine-tuning the model on a particular domain because (1) you do not need a large collection of documents to fine-tune on and (2) you don't need the compute resources to train an extremely large model. Instead, just write a small sample of target outputs (e.g., 20 hand written sentences) and run the attack in a few minutes on one GPU.
+ Use beam search for the optimization
+ Tune the size of the prepended token

Things that are left out that we found were not super necessary. Feel free to add these back:
+ Inside the inner loop sample a batch of racist tweets (or whatever content you'd like) and optimize over it.
+ Sample highly frequent n-grams as the "user input" and optimize the prepended token. This allows the attack to be universal over "any input".
