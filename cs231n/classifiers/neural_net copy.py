from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    z = np.dot(X, W1) + b1 #layer1 계산 
    h = np.maximum(z,0) #Relu
    scores = np.dot(h, W2) + b2 #layer2 계산

    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None

    sfm = np.exp(scores) #softmax 함수 사용 위해 각 score에 exp
    sfm /= np.sum(sfm, axis=1).reshape(N, 1) #각 exp의 합으로 나누어줌. axis=1 : 행 내 열끼리 합, reshape(N, 1) : N*1 배열로 변경

    loss = -np.sum(np.log(sfm[np.arange(N), y])) #log 취한 것 합해주고 batch size로 나눔. arrange:(0~n까지 하나씩.)
    loss /= N

    loss += 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2)) #Ls regularization

    # Backward pass: compute gradients
    grads = {}

    dsfm = np.copy(sfm) #(N, C)
    dsfm[np.arange(N), y] -= 1
    dh = np.dot(dsfm, W2.T)
    dz = dh * (z > 0) #(N, H)

    grads['W2'] = np.dot(h.T, dsfm) / N # (H, C)
    grads['b2'] = np.sum(dsfm, axis=0) / N # (C, )
    grads['W1'] = np.dot(X.T, dz) / N # (D, H)
    grads['b1'] = np.sum(dz, axis=0) / N #(H, )

    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      random_idxs = np.random.choice(num_train, batch_size)
      X_batch = X[random_idxs]
      y_batch = y[random_idxs]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']

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
    y_pred = None

    z = np.dot(X, self.params['W1']) + self.params['b1']
    h = np.maximum(z, 0)
    out = np.dot(h, self.params['W2']) + self.params['b2']
    y_pred = np.argmax(out, axis=1)

    return y_pred


