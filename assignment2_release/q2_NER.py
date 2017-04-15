import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn

from sklearn.model_selection import train_test_split



class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  embed_size = 50
  batch_size = 128
  label_size = 5
  hidden_size = [80, 80]
  max_epochs = 50
  early_stopping = 10
  dropout = 0.7
  lr = 0.5e-5
  l2 = 1e-4
  window_size = 3


class NERModel(LanguageModel):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.
  """

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    self.wv, word_to_num, num_to_word = ner.load_wv(
      'data/ner/vocab.txt', 'data/ner/wordVectors.txt')
    tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
    self.num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in self.num_to_tag.iteritems()}

    # Load the training set
    docs = du.load_dataset('data/ner/train')
    self.X_train, self.y_train = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]

    # Load the dev set (for tuning hyperparameters)
    docs = du.load_dataset('data/ner/dev')
    self.X_dev, self.y_dev = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    docs = du.load_dataset('data/ner/test.masked')
    self.X_test, self.y_test = du.docs_to_windows(
        docs, word_to_num, tag_to_num, wsize=self.config.window_size)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE

    self.input_placeholder = tf.placeholder(dtype=tf.int32,
                                            shape=(None, self.config.window_size),
                                            name='input')

    self.labels_placeholder = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.config.label_size),
                                             name='labels')

    self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())


    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE

    feed_dict = {
        self.input_placeholder: input_batch,
        self.dropout_placeholder: dropout,
    }

    if label_batch is not None:
        feed_dict[self.labels_placeholder] = label_batch

    return feed_dict

  def add_embedding(self):
    """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    (Words are discrete entities. They need to be transformed into vectors for use
    in deep-learning. Although we won't do so in this problem, in practice it's
    useful to initialize the embedding with pre-trained word-vectors. For this
    problem, using the default initializer is sufficient.)

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: See following link to understand what -1 in a shape means.
      https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.wv), embed_size)

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      retrived_emebdding = tf.nn.embedding_lookup(self.wv, self.input_placeholder)
      window = tf.reshape(retrived_emebdding, shape=(-1, self.config.window_size * self.config.embed_size))
      ### END YOUR CODE
      return window

  def add_model(self, window):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, while
          dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#graph-collections
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """

    window = tf.cast(window, tf.float32)

    with tf.variable_scope('Input_Layer') as scope:
        shape = (self.config.window_size * self.config.embed_size, self.config.hidden_size[0])
        # self.W1 = tf.get_variable('w1', shape=shape, initializer=xavier_weight_init())

        self.W1 = tf.Variable(xavier_weight_init()(shape).initialized_value())
        self.b1 = tf.Variable(tf.zeros(shape=(self.config.hidden_size[0],), dtype=tf.float32))
        # self.b1 = tf.get_variable('b1', shape=(self.config.hidden_size[0]), initializer=tf.constant_initializer(0.0))

    layer_output_1 = tf.tanh(tf.matmul(window, self.W1) + self.b1)
    layer_output_1 = tf.nn.dropout(layer_output_1, keep_prob=self.config.dropout)

    with tf.variable_scope('hidden_1') as scope:
        shape = (self.config.hidden_size[0], self.config.hidden_size[1])
        self.W2 = tf.Variable(xavier_weight_init()(shape).initialized_value())
        self.b2 = tf.Variable(tf.zeros(shape=(self.config.hidden_size[1]), dtype=tf.float32))

    #layer_output_2 = tf.nn.relu(tf.matmul(layer_output_1, self.W2) + self.b2)
    #layer_output_2 = tf.nn.dropout(layer_output_2, keep_prob=self.config.dropout)

    # with tf.variable_scope('hidden_2') as scope:
    #     shape = (self.config.hidden_size[1], self.config.hidden_size[2])
    #     self.W3 = tf.Variable(xavier_weight_init()(shape).initialized_value())
    #     self.b3 = tf.Variable(tf.zeros(shape=(self.config.hidden_size[2]), dtype=tf.float32))
    #
    # layer_output_3 = tf.tanh(tf.matmul(layer_output_2, self.W3) + self.b3)
    # layer_output_3 = tf.nn.dropout(layer_output_3, keep_prob=self.config.dropout)

    # with tf.variable_scope('hidden_4') as scope:
    #     shape = (self.config.hidden_size[2], self.config.hidden_size[3])
    #     self.W3 = tf.Variable(xavier_weight_init()(shape).initialized_value())
    #     self.b3 = tf.Variable(tf.zeros(shape=(self.config.hidden_size[2]), dtype=tf.float32))
    #
    # layer_output_3 = tf.nn.relu(tf.matmul(layer_output_2, self.W3) + self.b3)
    # layer_output_3 = tf.nn.tanh(layer_output_3)

    with tf.variable_scope('Softmax') as softmax_scope:
        u_shape = (self.config.hidden_size[0], self.config.label_size)
        # self.U = tf.get_variable('u', shape=u_shape, initializer=xavier_weight_init().init)
        # self.b3 = tf.get_variable('b3', shape=[self.config.label_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        self.U = tf.Variable(xavier_weight_init()(u_shape).initialized_value())
        self.b3 = tf.Variable(tf.zeros(shape=(self.config.label_size)))

    output = tf.nn.softmax(tf.matmul(layer_output_1, self.U) + self.b3)

    loss_paramters = [self.W1, self.b1, self.b3, self.U]

    l2_loss = sum(map(lambda p: tf.nn.l2_loss(p), loss_paramters))

    tf.add_to_collection(name='l2_loss', value=l2_loss)

    ### END YOUR CODE
    return output 

  def add_loss_op(self, y):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=y))

    loss += self.config.l2 * tf.get_collection('l2_loss')[0]

    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE

    global_step = tf.Variable(0, trainable=True)
    starter_learning_rate = self.config.lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.96, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss=loss, global_step=global_step)
    ### END YOUR CODE
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    window = self.add_embedding()
    y = self.add_model(window)

    self.loss = self.add_loss_op(y)
    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)

  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size, shuffle=shuffle)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results

def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print
    print confusion
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)

def calculate_confusion(config, predicted_indices, y_indices):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write(str(prediction) + "\n")

def NER_dev():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """


  train_loss_his = []
  train_acc_his = []
  valid_acc_his = []
  valid_loss_his = []

  train_hist = {
      'train_loss': train_loss_his,
      'train_acc': train_acc_his,
      'valid_loss': valid_loss_his,
      'valid_acc': valid_acc_his

  }


  with tf.Graph().as_default():

    config = Config()
    ratio = 0.3  # train ratio, set small when test hyper tuning

    model = NERModel(config)
    total_train_num = int(len(model.X_train) * ratio)
    dev_data_set_indices = np.random.choice(range(len(model.X_train)), total_train_num, replace=True)

    X_train, X_test, y_train, y_test = train_test_split(model.X_train[dev_data_set_indices],
                                                          model.y_train[dev_data_set_indices],
                                                          test_size=0.25, random_state=0)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###p

        train_loss, train_acc = model.run_epoch(session, X_train,
                                                y_train)

        val_loss, predictions = model.predict(session, X_test, y_test)

        val_acc = np.mean(predictions == y_test)

        print 'Training loss: {}'.format(train_loss)
        print 'Training acc: {}'.format(train_acc)
        print 'Validation loss: {}'.format(val_loss)
        print 'Validation acc: {}'.format(val_acc)

        train_loss_his.append(train_loss)
        train_acc_his.append(train_acc)
        valid_loss_his.append(val_loss)
        valid_acc_his.append(val_acc)

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")

          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
#        confusion = calculate_confusion(config, predictions, model.y_dev)
 #       print_confusion(confusion, model.num_to_tag)
        print 'Total time: {}'.format(time.time() - start)
      
      saver.restore(session, './weights/ner.weights')
      print 'Test'
      print '=-=-='
      print 'Writing predictions to q2_test.predicted'
   #   _, predictions = model.predict(session, model.X_test, model.y_test)
   #   save_predictions(predictions, "q2_test.predicted")

      return train_hist


def plot_train(train_hist):
    plt.subplot(2, 1, 1)
    train_acc = train_hist['train_acc']
    valid_acc = train_hist['valid_acc']
    x = range(len(train_acc))
    plt.plot(x, train_acc, 'g-')
    plt.plot(x, valid_acc, 'b-')
    plt.title('Train(Green) and Valid(Blue) acc')


    train_loss = train_hist['train_loss']
    valid_loss = train_hist['valid_loss']
    plt.subplot(2, 1, 2)
    plt.plot(x, train_loss, 'g-')
    plt.plot(x, valid_loss, 'b-')
    plt.title('Train(Green) and Valid(Blue) loss')

    plt.show()

if __name__ == "__main__":
    train_hist = NER_dev()
    plot_train(train_hist)

