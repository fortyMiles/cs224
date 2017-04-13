from unittest import TestCase
import numpy as np
import tensorflow as tf
from q1_softmax import cross_entropy_loss


class TestCross_entropy_loss(TestCase):
    def test_cross_entropy_loss_basic(self):
        """
        Some simple tests to get you started.
        Warning: these are not exhaustive.
        """
        y = np.array([[0, 1], [1, 0], [1, 0]])
        yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

        test1 = cross_entropy_loss(
            tf.convert_to_tensor(y, dtype=tf.int32),
            tf.convert_to_tensor(yhat, dtype=tf.float32))
        with tf.Session():
            test1 = test1.eval()
        result = -3 * np.log(.5)
        assert np.amax(np.fabs(test1 - result)) <= 1e-6
        print "Basic (non-exhaustive) cross-entropy tests pass\n"
