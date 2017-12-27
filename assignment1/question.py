      def svm_loss_vectorized(W, X, y, reg):
        """
        Structured SVM loss function, vectorized implementation.

        Inputs and outputs are the same as svm_loss_naive.
        """
        loss = 0.0
        dW = np.zeros(W.shape) # initialize the gradient as zero
        # transpose X and W
        # X.shape will be (D,N)
        # W.shape will be (C,D)
        X = X.T
        W = W.T
        dW = dW.T
        
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
        for i,label in enumerate(y):
          dW[label] -= dW_y[i]

        loss = np.sum(margins*mask) - num_train
        loss /= num_train
        dW /= num_train
        # add regularization term
        loss += reg * np.sum(W*W)
        dW += reg * 2 * W
        dW = dW.T
        
        return loss, dW