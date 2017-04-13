import numpy as np
import tensorflow as tf

def softmax(x):
  """
  Compute the softmax function in tensorflow.

  You might find the tensorflow functions tf.exp, tf.reduce_max,
  tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
  not need to use all of these functions). Recall also that many common
  tensorflow operations are sugared (e.g. x * y does a tensor multiplication
  if x and y are both tensors). Make sure to implement the numerical stability
  fixes as in the previous homework!

  Args:
    x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
         represented by row-vectors. (For simplicity, no need to handle 1-d
         input as in the previous homework)
  Returns:
    out: tf.Tensor with shape (n_sample, n_features). You need to construct this
         tensor in this problem.
  """

  ### YOUR CODE HERE

  per_iter_max = tf.reduce_max(x, axis=1, keep_dims=True)
  y = tf.subtract(x, per_iter_max)
  exp_matrix = tf.exp(y)
  exp_sum = tf.reduce_sum(exp_matrix, axis=1, keep_dims=True, name='get_exp_sum')
  out = tf.divide(exp_matrix, exp_sum)

  return out


def cross_entropy_loss(y, yhat):
  """
  Compute the cross entropy loss in tensorflow.

  y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
  of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
  be of dtype tf.float32.

  The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
  solutions are possible, so you may not need to use all of these functions).

  Note: You are NOT allowed to use the tensorflow built-in cross-entropy
        functions.

  Args:
    y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
    yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
          probability distribution and should sum to 1.
  Returns:
    out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
          tensor in the problem.
  """
  ### YOUR CODE HERE

  out = -tf.reduce_sum(tf.cast(y, dtype=tf.float32) * tf.log(yhat), axis=0)
  out = tf.reduce_sum(out)
  ### END YOUR CODE
  return out

