import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = (X[i].dot(W)).reshape((-1,1))
    exp_scores_shifted = np.exp(scores - np.max(scores))

    y_pred = (exp_scores_shifted/np.sum(exp_scores_shifted))
    loss += -np.log(y_pred[y[i]])

    dW += np.dot(y_pred, X[i].reshape(1,-1)).T
    dW[:,y[i]] -= X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W*W)
  dW += 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_classes = W.shape[1]

  scores = X.dot(W)
  exp_scores_shifted = np.exp(scores - np.max(scores,axis=-1,keepdims=True))

  y_pred = (exp_scores_shifted/np.sum(exp_scores_shifted,axis=-1,keepdims=True)).T
  
  mask = y == np.arange(num_classes)[:,None]
  true_probabilities = np.sum(mask * y_pred,axis=0)
  loss += np.sum(-np.log(true_probabilities))


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += reg * np.sum(W * W)

  return loss, dW

