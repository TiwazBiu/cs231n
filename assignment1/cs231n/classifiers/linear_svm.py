import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:]
        loss += margin


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  ############################################################################# 
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (D, N) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # transpose  W,X
  # W.shape will be (C,D)
  # X.shape will be (N,D)
  W = W.T
  X = X.T
  dW = dW.T
  num_classes = W.shape[0]
  num_dims = X.shape[0]
  num_train = X.shape[1]
  # W_y  shape from (N,D) to (D,N) 
  W_y = W[y].T
  S_y = np.sum(W_y*X ,axis=0)
  margins =  np.dot(W,X) + 1 - S_y
  mask = np.array(margins>0)

  # get the value of num_train examples made on W's gradient
  # that is,only when the mask is positive 
  # the train example has impact on W's gradient
  dW_j = np.dot(mask, X.T)
  dW +=  dW_j
  mul_mask = np.sum(mask, axis=0, keepdims=True).T
  
  # dW[y] -= mul_mask * X.T
  dW_y =  mul_mask * X.T
  # for i,label in enumerate(y):
  #   dW[label] -= dW_y[i]
  # np.subtract.at(dW,y,dW_y)
  y_mask = y == np.arange(num_classes)[:,None] 
  dW -= y_mask.dot(dW_y)

  loss = np.sum(margins*mask) - num_train
  loss /= num_train
  dW /= num_train
  # add regularization term
  loss += reg * np.sum(W*W)
  dW += reg * 2 * W
  dW = dW.T
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
