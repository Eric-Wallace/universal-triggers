model-name:
ifeq ($(MODEL_NAME),)
	echo "Please set MODEL_NAME before training (e.g., MODEL_NAME=test)"; exit 2;
endif

trigger:
ifeq ($(TRIGGER),)
	echo "Please set TRIGGER before training (e.g., TRIGGER=bad)"; exit 2;
endif

sampling:
	$(eval NUM_SEQ=100)

create-gpt2: 
	python create_adv_token.py \
		--model_type gpt2 \
		--model_name_or_path gpt2 

create-ctrl:
	python create_adv_token.py \
		--model_type ctrl \
		--model_name_or_path ctrl \
		--temperature 0.2 \
		--repetition_penalty 1.2

create-openai-gpt:
	python create_adv_token.py \
		--model_type openai-gpt \
		--model_name_or_path openai-gpt

create-xlnet:
	python create_adv_token.py \
		--model_type xlnet \
		--model_name_or_path xlnet-base-cased 

create-transfo-xl:
	python create_adv_token.py \
		--model_type transfo-xl \
		--model_name_or_path transfo-xl

create-xlm:
	python create_adv_token.py \
		--model_type xlm \
		--model_name_or_path xlm

create-distilgpt2: 
	python create_adv_token.py \
		--model_type distilgpt2 \
		--model_name_or_path distilgpt2

create-gptneo: 
	python create_adv_token.py \
		--model_type gptneo \
		--model_name_or_path "EleutherAI/gpt-neo-125M"

sample-gpt2: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type gpt2 \
		--model_name_or_path gpt2 \
		--trigger "$(TRIGGER)"

sample-ctrl: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type ctrl \
		--model_name_or_path ctrl \
		--repetition_penalty 1.2 \
		--temperature 0.2 \ 
		--trigger "$(TRIGGER)"

sample-openai-gpt: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type openai-gpt \
		--model_name_or_path openai-gpt \
		--trigger "$(TRIGGER)"

sample-xlnet: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type xlnet \
		--model_name_or_path xlnet-base-cased \
		--trigger "$(TRIGGER)"

sample-transfo-xl: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type transfo-xl \
		--model_name_or_path transfo-xl \
		--trigger "$(TRIGGER)"

sample-xlm: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type xlm \
		--model_name_or_path xlm \
		--trigger "$(TRIGGER)"

sample-distilgpt2: sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type distilgpt2 \
		--model_name_or_path distilgpt2 \
		--trigger "$(TRIGGER)"

sample-gptneo:  sampling trigger
	python sample_from_model.py \
		--num_return_sequences $(NUM_SEQ) \
		--model_type gptneo \
		--model_name_or_path "EleutherAI/gpt-neo-125M" \
		--trigger "$(TRIGGER)"
