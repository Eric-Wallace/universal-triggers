from operator import itemgetter
from copy import deepcopy
import heapq
import numpy
import torch
import torch.optim as optim
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import move_to_device
from allennlp.modules.text_field_embedders import TextFieldEmbedder

def get_embedding_weight(model):
    """
    Extracts and returns the token embedding weight matrix from the model.
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                embedding_weight = module._token_embedders[embed].weight.cpu().detach()
    return embedding_weight

# hook used in add_hooks()
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def add_hooks(model):
    """
    Finds the token embedding matrix on the model and registers a hook onto it.
    When loss.backward() is called, extracted_grads list will be filled with
    the gradients w.r.t. the token embeddings
    """
    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            for embed in module._token_embedders.keys():
                module._token_embedders[embed].weight.requires_grad = True
            module.register_backward_hook(extract_grad_hook)

def evaluate_batch(model, batch, trigger_token_ids=None, snli=False):
    """
    Takes a batch of classification examples (SNLI or SST), and runs them through the model.
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    batch = move_to_device(batch[0], cuda_device=0)
    if trigger_token_ids is None:
        if snli:
            model(batch['premise'], batch['hypothesis'], batch['label'])
        else:
            model(batch['tokens'], batch['label'])
        return None
    else:
        trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch['label']), 1).cuda()
        if snli:
            original_tokens = batch['hypothesis']['tokens'].clone()
            batch['hypothesis']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
            output_dict = model(batch['premise'], batch['hypothesis'], batch['label'])
            batch['hypothesis']['tokens'] = original_tokens
        else:
            original_tokens = batch['tokens']['tokens'].clone()
            batch['tokens']['tokens'] = torch.cat((trigger_sequence_tensor, original_tokens), 1)
            output_dict = model(batch['tokens'], batch['label'])
            batch['tokens']['tokens'] = original_tokens
        return output_dict

def get_average_grad(model, batch, trigger_token_ids, target_label=None, snli=False):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch[0]['label'].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch[0]['label'] = int(target_label) * torch.ones_like(batch[0]['label']).cuda()
    global extracted_grads
    extracted_grads = [] # clear existing stored grads
    loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss']
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()
    batch[0]['label'] = original_labels # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)] # return just trigger grads
    return averaged_grad

def get_accuracy(model, dev_dataset, vocab, trigger_token_ids=None, snli=False):
    """
    When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
    triggers prepended for the whole dev_dataset.
    """
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    if snli:
        iterator = BucketIterator(batch_size=128, sorting_keys=[("premise", "num_tokens")])
    else:
        iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    if trigger_token_ids is None:
        for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print("Without Triggers: " + str(model.get_metrics()['accuracy']))
    else:
        print_string = ""
        for idx in trigger_token_ids:
            print_string = print_string + vocab.get_token_from_index(idx) + ', '

        for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
            evaluate_batch(model, batch, trigger_token_ids, snli)
        print("Current Triggers: " + print_string + " : " + str(model.get_metrics()['accuracy']))

def get_best_candidates(model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids, snli)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, cand,
                                                             cand_trigger_token_ids, snli))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids, snli=False):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    model.get_metrics(reset=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss = evaluate_batch(model, batch, trigger_token_ids, snli)['loss'].cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id] # replace one token
        loss = evaluate_batch(model, batch, trigger_token_ids_one_replaced, snli)['loss'].cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate
