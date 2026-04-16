# from scipy.misc import imread, imresize

import json
import os
from collections import Counter
from random import seed, choice, sample

import h5py
import numpy as np
import torch
from tqdm import tqdm

from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.rouge.rouge import Rouge


# from eval_func.spice.spice import Spice
# from scipy.misc import imread, imresize
# from skimage.transform import resize

from eval_func.spice.spice import Spice

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(args, data_name, epoch, epochs_since_improvement,
                    encoder_image_ms, encoder_feat, decoder,
                    encoder_image_ms_optimizer,
                    encoder_feat_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint for dual-encoder setup.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param encoder_image_ms: encoder for Sentinel-2
    :param encoder_image_ms: encoder for Sentinel-1
    :param encoder_feat: fusion module
    :param decoder: decoder model
    :param encoder_image_ms_optimizer: optimizer for MS encoder
    :param encoder_feat_optimizer: optimizer for fusion
    :param decoder_optimizer: optimizer for decoder
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder_image': encoder_image_ms,
        'encoder_feat': encoder_feat,
        'decoder': decoder,
        'encoder_image_optimizer': encoder_image_ms_optimizer,
        'encoder_feat_optimizer': encoder_feat_optimizer,
        'decoder_optimizer': decoder_optimizer,
    }

    filename = 'checkpoint_' + data_name + '.pth.tar'
    path = args.savepath

    if not os.path.exists(path):
        os.makedirs(path)

    if is_best:
        torch.save(state, os.path.join(path, 'NEW_BEST_' + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score2(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def get_eval_score(references, hypotheses, word_map):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(word_map), "SPICE")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

def get_eval_score_detailed(references, hypotheses, word_map, return_per_sample: bool = False, verbose: bool = True):

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        #(Spice(word_map), "SPICE"),
    ]

    # stringify tokens (same logic as before)
    hypo = [[' '.join(h)] for h in [[str(x) for x in h] for h in hypotheses]]
    ref  = [[' '.join(r) for r in refs] for refs in
            [[[str(x) for x in r] for r in refs] for refs in references]]

    overall = {}
    per_img = [dict() for _ in range(len(hypotheses))] if return_per_sample else None

    for scorer, names in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)

        if verbose:
            print(f"{names} {score_i}")

        if isinstance(names, list):
            # BLEU-1..4: score_i is list[4], scores_i is list[4][N]
            for j, n in enumerate(names):
                overall[n] = float(score_i[j])
                if return_per_sample:
                    for idx, v in enumerate(scores_i[j]):
                        per_img[idx][n] = float(v)
        else:
            overall[names] = float(score_i if not isinstance(score_i, list) else score_i[0])
            if return_per_sample:
                for idx, v in enumerate(scores_i):
                    per_img[idx][names] = float(v)

    return (overall, per_img) if return_per_sample else overall

def convert2words(sequences, rev_word_map):
    for l1 in sequences:
        caption = ""
        for l2 in l1:
            caption += rev_word_map[l2]
            caption += " "
        print(caption)
