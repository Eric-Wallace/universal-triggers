from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir='./.cache')
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

m = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", cache_dir='./.cache')

prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
