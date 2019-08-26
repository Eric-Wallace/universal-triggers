import sys
from copy import deepcopy
from operator import itemgetter
import heapq
import torch
import torch.optim as optim
import numpy
from allennlp.data.tokenizers import WordTokenizer
from allennlp.training.metrics import SquadEmAndF1
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import move_to_device
tokenizer = WordTokenizer()
sys.path.append('..')
import utils

def evaluate_batch_squad(model, batch, trigger_token_ids, vocab, span_start, span_end):
    """
    Same as evaluate_batch() in utils.py but handles modifying SQuAD paragraphs and answer spans.
    """
    batch = move_to_device(batch[0], cuda_device=0)

    # convert ids to words for using in adding to the metadata for SQuAD evaluation script
    trigger_words = []
    for idx in trigger_token_ids:
        trigger_words.append(vocab.get_token_from_index(idx))
    assert(len(trigger_token_ids)) == len(trigger_words)

    trigger_sequence_tensor = torch.LongTensor(trigger_token_ids).repeat(batch['passage']['tokens'].shape[0], 1).cuda()
    original_tokens = batch['passage']['tokens'].clone()
    # append trigger to front of original passage tokens
    batch['passage']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)

    # spans are set to the spot where the target inside the trigger is
    batch['span_start'] = torch.LongTensor([span_start]).repeat(batch['passage']['tokens'].shape[0], 1).cuda()
    batch['span_end'] = torch.LongTensor([span_end]).repeat(batch['passage']['tokens'].shape[0], 1).cuda()

    # append the triggers to the metadata so you can compute the F1 and such correctly.
    orig_original_passage = []
    orig_passage_tokens = []
    orig_token_offsets = []
    for idx in range(len(batch['metadata'])):
        # copy the original metadata
        orig_original_passage.append(deepcopy(batch['metadata'][idx]['original_passage']))
        orig_passage_tokens.append(deepcopy(batch['metadata'][idx]['passage_tokens']))
        orig_token_offsets.append(deepcopy(batch['metadata'][idx]['token_offsets']))
        # add triggers to metadata
        batch['metadata'][idx]['original_passage'] = ' '.join(trigger_words) + " " + batch['metadata'][idx]['original_passage']
        batch['metadata'][idx]['passage_tokens'] = trigger_words + batch['metadata'][idx]['passage_tokens']

        # update passage_offsets
        passage_tokens = tokenizer.tokenize(batch['metadata'][idx]['original_passage'])
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        batch['metadata'][idx]['token_offsets'] = passage_offsets

    output_dict = model(batch['question'], batch['passage'], batch['span_start'], batch['span_end'], batch['metadata'])
    # reset metadata and tokens
    for idx, item in enumerate(batch['metadata']):
        batch['metadata'][idx]['original_passage'] = orig_original_passage[idx]
        batch['metadata'][idx]['passage_tokens'] = orig_passage_tokens[idx]
        batch['metadata'][idx]['token_offsets'] = orig_token_offsets[idx]
    batch['passage']['tokens'] = original_tokens

    return output_dict

def get_accuracy_squad(model, dev_dataset, vocab, trigger_token_ids, answer, span_start, span_end):
    """
    Same as get_accuracy() in utils.py but for SQuAD models.
    """
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=32, sorting_keys=[["passage", "num_tokens"], ["question", "num_tokens"]])
    iterator.index_with(vocab)

    # Print out the current triggers.
    print_string = ""
    trigger_words = []
    for idx in trigger_token_ids:
        print_string = print_string + vocab.get_token_from_index(idx) + ', '
        trigger_words.append(vocab.get_token_from_index(idx))
    print("Current Triggers: " + print_string)

    # Evaluate the model using the triggers and get the F1 / EM scores with the target.
    total_f1 = 0.0
    total_em = 0.0
    total = 0.0
    for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
        torch.cuda.empty_cache()  # TODO may be unnecessary but sometimes memory caching cuases OOM
        output_dict = evaluate_batch_squad(model, batch, trigger_token_ids, vocab, span_start, span_end)
        # go through the model's predictions and compute F1 and EM with the target span.
        for span_str in output_dict['best_span_str']:
            metrics = SquadEmAndF1()
            metrics.get_metric(reset=True)
            metrics(span_str, [answer])
            em, f1 = metrics.get_metric()
            total_f1 += f1
            total_em += em
            total += 1.0

    print("F1 with target span: " + str(total_f1 / total))
    print("EM with target span: " + str(total_em / total))

def get_average_grad_squad(model, vocab, trigger_token_ids, dev_dataset, span_start, span_end):
    """
    Same as get_average_grad() in utils.py, except that we use the entire development set to
    compute the gradient for the triggers (rather than one batch).
    """
    batch_count = 0
    optimizer = optim.Adam(model.parameters())
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=True), group_size=1):
        optimizer.zero_grad()
        utils.extracted_grads = [] # clear existing stored grads
        loss = evaluate_batch_squad(model, batch, trigger_token_ids, vocab, span_start, span_end)['loss']
        loss.backward()
        if batch_count == 0:
            grads = torch.sum(utils.extracted_grads[0], dim=0).detach()[0:len(trigger_token_ids)] # inddex 0 is passage
        else:
            grads += torch.sum(utils.extracted_grads[0], dim=0).detach()[0:len(trigger_token_ids)]
        batch_count = batch_count + 1

    averaged_grad = grads / batch_count
    return averaged_grad.cpu()

def get_best_candidates_squad(model, trigger_token_ids, cand_trigger_token_ids, vocab, dev_dataset,
                              beam_size, ignore_indices, span_start, span_end):
    """
    Follows get_best_candidates() in utils.py except it assumes a targeted loss, takes in the span start/end.
    """
    loss_per_candidate = get_loss_per_candidate_squad(0, model, trigger_token_ids, cand_trigger_token_ids,
                                                      vocab, dev_dataset, span_start, span_end)
    top = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
    for idx in range(1, len(trigger_token_ids)):
        if ignore_indices is not None and ignore_indices[idx] == 1:
            continue
        loss_per_candidate = []
        for cand, _ in top: # for all the candidates in the beam
            loss_per_candidate.extend(get_loss_per_candidate_squad(idx, model, cand, cand_trigger_token_ids, vocab,
                                                                   dev_dataset, span_start, span_end))
    top = heapq.nsmallest(beam_size, loss_per_candidate, key=itemgetter(1))
    return min(top, key=itemgetter(1))[0]

def get_loss_per_candidate_squad(index, model, trigger_token_ids, cand_trigger_token_ids, vocab,
                                 dev_dataset, span_start, span_end):
    """
    Similar to get_loss_per_candidate, except that we use multiple batches (in this case 4) rhater than one
    to evaluate the top trigger token candidates.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    model.get_metrics(reset=True)
    loss_per_candidate = []
    iterator = BasicIterator(batch_size=32)
    batch_count = 0
    curr_loss = 0.0
    for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=True), group_size=1):
        if batch_count > 4:
            continue
        batch_count = batch_count + 1
        curr_loss += evaluate_batch_squad(model, batch, trigger_token_ids,
                                          vocab, span_start, span_end)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))

    for cand_id in range(len(cand_trigger_token_ids[0])):
        temp_trigger_token_ids = deepcopy(trigger_token_ids)
        temp_trigger_token_ids[index] = cand_trigger_token_ids[index][cand_id]
        loss = 0
        batch_count = 0
        for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=True), group_size=1):
            if batch_count > 4:
                continue
            batch_count = batch_count + 1
            loss += evaluate_batch_squad(model, batch, temp_trigger_token_ids,
                                         vocab, span_start, span_end)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(temp_trigger_token_ids), loss))
    return loss_per_candidate
