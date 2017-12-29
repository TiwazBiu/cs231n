from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout_rate = 1.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    hidden_size = W1.shape[1]
    if dropout_rate < 1.0: 
      hidden_mask = np.random.choice([0,1],hidden_size, \
                        p=[1.0-dropout_rate,dropout_rate])
    else:
      hidden_mask = np.ones(hidden_size)
    masked_W1 = hidden_mask*W1
    masked_b1 = hidden_mask*b1
    # Compute the forward pass
    
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    z1 = X.dot(masked_W1) + masked_b1
    mask_z1 = z1>0
    a1 = mask_z1*z1
    z2 = a1.dot(W2) + b2
    
    scores = z2
    num_classes = scores.shape[1]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # z1.shape==(N,H)
    # z1 = X.dot(W1) + b1  
    # mask = z1>0
    # a1 = mask*z1
    
    # z2.shape==(N,C)
    # z2 = a1.dot(W2) + b2
    # z2max.shape==(N,1)
    z2_max = np.max(z2,axis=1,keepdims=True)
    z2_shifted = z2 - z2_max
    z2_exp = np.exp(z2_shifted)
    
    # den.shape==(N,1)
    den = np.sum(z2_exp,axis=1,keepdims=True)
    inv_den = 1.0/den

    # y_pred.shape==(N,C)
    y_pred = z2_exp*inv_den

    # mask.shape==(C,N)
    mask_y = y == np.arange(num_classes)[:,None]
    # mask_y.shape==(N,C)
    y_masked = mask_y.T * y_pred

    # py.shape==(N,1)
    py = np.sum(y_masked,axis=1)
    log_py = -np.log(py)

    loss += np.sum(log_py)/N
    loss += reg*(np.sum(W1*W1)+np.sum(W2*W2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    grads['W1'] = np.zeros_like(W1)
    grads['b1'] = np.zeros_like(b1)
    grads['W2'] = np.zeros_like(W2)
    grads['b2'] = np.zeros_like(b2)

    grads['W1'] += 2*reg*W1
    grads['W2'] += 2*reg*W2
    
    dlog_py = np.ones_like(log_py) * 1/N
    dpy = (dlog_py * (-1.0/py)).reshape(-1,1)
    # print(dpy.shape)
    dy_masked = np.tile(dpy,(1,num_classes))
    # print(dmask_y.shape)
    # dy_pred==(N,C)
    dy_pred = dy_masked * mask_y.T
    # print(dy_masked.shape)
    dz2_exp = dy_pred * inv_den
    dinv_den = np.sum(dy_pred*z2_exp,axis=1,keepdims=True)
    dden = dinv_den * (-1.0/(den*den))
    dz2_exp += np.tile(dden,(1,num_classes))

    # dz2.shape==(N,C)
    dz2 = dz2_exp * z2_exp

    grads['W2'] += a1.T.dot(dz2)
    grads['b2'] += np.sum(dz2,axis=0)

    da1 = dz2.dot(W2.T)
    dz1 = da1*mask_z1
    grads['W1'] += X.T.dot(dz1)
    grads['b1'] += np.sum(dz1,axis=0)

    grads['W1'] *= hidden_mask
    grads['b1'] *= hidden_mask

    
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, dropout_rate = 1.0, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = int(max(num_train / batch_size, 1))
    print('iterations_per_epoch : ',iterations_per_epoch)
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = np.random.randint(num_train,size=batch_size)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, \
                                dropout_rate=dropout_rate)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['b2'] -= learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    z1 = X.dot(W1) + b1  
    mask = z1>0
    a1 = mask*z1
    z2 = a1.dot(W2) + b2
    z2_max = np.max(z2,axis=1,keepdims=True)
    z2_shifted = z2 - z2_max
    num = np.exp(z2_shifted)
    den = np.sum(num,axis=1,keepdims=True)

    # y_pred.shape==(N,C)
    y_pred = num/den
    y_pred = np.argmax(y_pred,axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


