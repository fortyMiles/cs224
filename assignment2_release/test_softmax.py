from unittest import TestCase
from q1_softmax import softmax
import tensorflow as tf
import numpy as np


class TestSoftmax(TestCase):

    def test_softmax_basic(self):
      """
      Some simple tests to get you started. 
      Warning: these are not exhaustive.
      """
      print "Running basic tests..."
      test1 = softmax(tf.convert_to_tensor(
          np.array([[1001,1002],[3,4]]), dtype=tf.float32))
      with tf.Session():
          test1 = test1.eval()

      self.assertTrue(np.amax(np.fabs(test1 - np.array(
          [0.26894142,  0.73105858]))) <= 1e-6)

    def test_softmax_2(self):
      test2 = softmax(tf.convert_to_tensor(
          np.array([[-1001,-1002]]), dtype=tf.float32))
      with tf.Session():
          test2 = test2.eval()

      self.assertTrue(np.amax(np.fabs(test2 - np.array(
          [0.73105858, 0.26894142]))) <= 1e-6)


      print "Basic (non-exhaustive) softmax tests pass\n"

