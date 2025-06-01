# import tensorflow as tf
# from protos import triplet_mining_pb2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def distance_fn(x, y):
      """Distance function."""
      keep_prob = True
      distance = nn.functional.dropout(torch.multiply(x, y), keep_prob,
          training=True)
      distance = 1 - torch.sum(distance, dim=1)
      return distance

def refine_fn(pred_image, pos_indices, neg_indices):
      """Refine function."""
      pos_ids = torch.gather(pred_image, pos_indices)
      neg_ids = torch.gather(pred_image, neg_indices)

      masks = torch.not_equal(pos_ids, neg_ids)
      pos_indices = torch.masked_select(pos_indices, masks)
      neg_indices = torch.masked_select(neg_indices, masks)
      return pos_indices, neg_indices


def _safe_batch_size(tensor):
  """Safely gets the batch size of tensor. 
  Args:
    tensor: a [batch, ...] tensor.
  Returns:
    batch_size: batch size of the tensor.
  """
  batch_size = tensor.shape()[0].value
  if batch_size is None:
    batch_size = (tensor).shape()[0]
  return batch_size


def _mine_all_examples(distances):
  """Mines all examples.
  Mine all returns all True examples in the following matrix:
  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances)
  indices = torch.where(torch.less(torch.diag(np.fill([batch_size], 1)), 1))
  return indices[:, 0], indices[:, 1]
  

def _mine_random_examples(distances, negatives_per_anchor):
  """Mines random batch examples.
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
    negatives_per_anchor: number of negatives per each anchor.
  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances) 

  pos_indices = torch.tile(torch.range(batch_size), [negatives_per_anchor])
  indices = (batch_size - 1) * torch.rand(pos_indices, dtype=torch.int32)
  neg_indices = np.mod(pos_indices + indices, batch_size)

  return pos_indices, neg_indices



def _mine_hard_examples(distances, top_k):
  """Mines hard examples.
  Mine hard returns examples with smallest values in the following masked matrix:
  / 0, 1, 1, 1 \
  | 1, 0, 1, 1 |
  | 1, 1, 0, 1 |
  \ 1, 1, 1, 0 /
    
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
    top_k: number of negative examples to choose per each row.
  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  batch_size = _safe_batch_size(distances)
  top_k = torch.minimum(top_k, batch_size - 1)
  

  pos_indices = torch.unsqueeze(torch.range(batch_size, dtype=torch.int32), 1)
  pos_indices = torch.tile(pos_indices, [1, 1 + top_k])

  _, neg_indices = torch.topk(-distances, k=1 + top_k)

  masks = torch.not_equal(pos_indices, neg_indices)
  pos_indices = torch.masked_select(pos_indices, masks)
  neg_indices = torch.masked_select(neg_indices, masks)

  return pos_indices, neg_indices


def _mine_semi_hard_examples(distances):
  """Mines semi-hard examples.
    
  Mine semi-hard returns examples that have dist_fn(a, p) < dist_fn(a, n).
  Args:
    distances: a [batch, batch] float tensor, in which distances[i, j] is the
      distance between i-th item and j-th item.
  Returns:
    pos_indices: a [batch] int64 tensor indicateing indices of positive examples.
    neg_indices: a [batch] int64 tensor indicateing indices of negative examples.
  """
  
  pos_distances = torch.unsqueeze(torch.diagonal(distances), 1)
  indices = torch.where(pos_distances < distances)
  return indices[:, 0], indices[:, 1]


def build_mining_func(config):
  """Builds triplet mining function based on config.
  Args:
    config: an instance of triplet_mining_pb2.TripletMining.
  Raises:
    ValueError if config is invalid.
  Returns:
    a callable that takes a distance matrix as input. 
  """
  triplet_mining = config.WhichOneof('triplet_mining')

  if 'mine_all' == triplet_mining:
    return _mine_all_examples

  if 'mine_semi_hard' == triplet_mining:
    return _mine_semi_hard_examples

  if 'mine_hard' == triplet_mining:
    top_k = config.mine_hard.top_k

    def _mine_hard_examples_wrap(distances):
      return _mine_hard_examples(distances, top_k)
    return _mine_hard_examples_wrap

  if 'mine_random' == triplet_mining:
    negatives_per_anchor = config.mine_random.negatives_per_anchor

    def _mine_random_examples_wrap(distances):
      return _mine_random_examples(distances, negatives_per_anchor)
    return _mine_random_examples_wrap

  raise ValueError('Invalid triplet_mining method {}.'.format(triplet_mining))


def compute_triplet_loss(anchors, positives, negatives, distance_fn, alpha):
  """Computes triplet loss.
  Args:
    anchors: a [batch, embedding_size] tensor.
    positives: a [batch, embedding_size] tensor.
    negatives: a [batch, embedding_size] tensor.
    distance_fn: a function using to measure distance between two [batch,
      embedding_size] tensors
    alpha: a float value denoting the margin.
  Returns:
    loss: the triplet loss tensor.
    summary: a dict mapping from summary names to summary tensors.
  """
  batch_size = _safe_batch_size(anchors)
  batch_size = torch.maximum(1e-12, batch_size.type(torch.float32))

  dist1 = distance_fn(anchors, positives)
  dist2 = distance_fn(anchors, negatives)

  losses = torch.maximum(dist1 - dist2 + alpha, 0)
  losses = torch.masked_select(losses, losses > 0)
  
  if losses.shape()[0] > 0:
    loss = losses.mean()
  else:
    loss = 0.0

  # Gather statistics.
  loss_examples = torch.count_nonzero(dist1 + alpha >= dist2)
  return loss, {'loss_examples': loss_examples}

def triplet_loss_wrap_func(
    anchors, positives, distance_fn, mining_fn, refine_fn, margin=float(0.1), tag=None):
  """Wrapper function for triplet loss.
  Args:
    anchors: a [batch, common_dimensions] tf.float32 tensor.
    positives: a [batch, common_dimensions] tf.float32 tensor.
    similarity_matrx: a [common_dimensions, common_dimensions] tf.float32 tensor.
    distance_fn: a callable that takes two batch of vectors as input.
    mining_fn: a callable that takes distance matrix as input.
    refine_fn: a callable that takes pos_indices and neg_indices as inputs.
    margin: margin alpha of the triplet loss.
  Returns:
    loss: the loss tensor.
  """
  distances = torch.multiply(
      torch.unsqueeze(anchors, 1), np.expand_dims(positives, 0))
  distances = 1 - torch.sum(distances, dim=2)

  pos_indices, neg_indices = mining_fn(distances)
  if not refine_fn is None:
    pos_indices, neg_indices = refine_fn(pos_indices, neg_indices)
    
  loss, summary = compute_triplet_loss(
      anchors=torch.gather(anchors, pos_indices), 
      positives=torch.gather(positives, pos_indices), 
      negatives=torch.gather(positives, neg_indices),
      distance_fn=distance_fn,
      alpha=margin)
      
  if tag is not None:
    writer = SummaryWriter()
    for k, v in summary.iteritems():
      writer.add_scalar('triplet_train/{}_{}'.format(tag, k), v)

  return loss, summary