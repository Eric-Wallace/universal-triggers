import sys
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import load_archive
from allennlp.data.iterators import BasicIterator
sys.path.append('..')
import utils
import squad_utils
import attacks

def main():
    # Read the SQuAD validation dataset using a word tokenizer
    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SquadReader(token_indexers={'tokens': single_id})
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json')
    # Load the model and its associated vocabulary.
    model = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-glove-2019.05.09.tar.gz').model
    vocab = model.vocab
    model.eval().cuda()

    # filter to just certain `wh` questions
    who_questions_dev, what_questions_dev, where_questions_dev, when_questions_dev, what_questions_dev, \
        how_questions_dev, why_questions_dev, which_questions_dev, other_questions_dev = ([] for i in range(9))
    for item in dev_dataset:
        for word in item['question']:
            if word.text.lower() == 'who':
                who_questions_dev.append(item)
                break
            if word.text.lower() == 'what':
                what_questions_dev.append(item)
                break
            if word.text.lower() == 'where':
                where_questions_dev.append(item)
                break
            if word.text.lower() == 'when':
                when_questions_dev.append(item)
                break
            if word.text.lower() == 'how':
                how_questions_dev.append(item)
                break
            if word.text.lower() == 'why':
                why_questions_dev.append(item)
                break
            if word.text.lower() == 'which':
                which_questions_dev.append(item)
                break
            else:
                other_questions_dev.append(item)

    # Use batches to craft the universal perturbations
    universal_perturb_batch_size = 32
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # We register a gradient hook on the embeddings.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # save the word embedding matrix

    # Initialize the trigger. The first one is an intialization with all "the" tokens.
    # You can customize it. Make sure to set the fixed target answer and the question type.
    # The second is a trigger found after running as reported in our paper.
    trigger_init = "the the the the donald trump the the the the"
    target_answer = "donald trump"
    subsampled_dev_dataset = who_questions_dev # universal attack on `who` questions
    # trigger_init = "why how ; known because : to kill american people ."
    # target_answer = "to kill american people"
    # subsampled_dev_dataset = why_questions_dev # universal attack on `who` questions

    # tokenizes the trigger, and finds the start/end span
    # make sure the trigger tokens are space separated
    trigger_token_ids = [vocab.get_token_index(t) for t in trigger_init.split(' ')]
    span_start = trigger_init.split(' ').index(target_answer.split(' ')[0]) # start of target_answer
    span_end = trigger_init.split(' ').index(target_answer.split(' ')[-1])
    # we ignore replacement at the positions of the answer (answer is fixed)
    ignore_indices = [0]*(span_start) + \
        [1]*(span_end - span_start + 1) + [0]*(len(trigger_token_ids) - 1 - span_end)

    # if these parameters are bigger = better result, but slower
    num_candidates = 20
    beam_size = 5
    for _ in range(100):
        # Get targeted accuracy
        squad_utils.get_accuracy_squad(model,
                                       subsampled_dev_dataset,
                                       vocab,
                                       trigger_token_ids,
                                       target_answer,
                                       span_start,
                                       span_end)
        model.train()

        # Get the gradient for the appended tokens averaged over the batch.
        averaged_grad = squad_utils.get_average_grad_squad(model,
                                                           vocab,
                                                           trigger_token_ids,
                                                           subsampled_dev_dataset,
                                                           span_start,
                                                           span_end)

        # Use an attack method to get the top candidates
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=num_candidates,
                                                        increase_loss=False)

        # Query the model with the top candidates to find the best tokens.
        trigger_token_ids = squad_utils.get_best_candidates_squad(model, trigger_token_ids,
                                                                  cand_trigger_token_ids,
                                                                  vocab,
                                                                  subsampled_dev_dataset,
                                                                  beam_size,
                                                                  ignore_indices,
                                                                  span_start,
                                                                  span_end)

if __name__ == '__main__':
    main()
