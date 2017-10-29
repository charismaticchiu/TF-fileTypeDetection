from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd



tf.logging.set_verbosity(tf.logging.INFO)

def prepare_file(filename):

  filename = 'temp_data.csv'
  data = pd.read_csv(filename)
  fts = data['0'].unique()   
  gdata = data.groupby('0')  
  grp = {}
  for ft in fts:
    grp[ft] = gdata.get_group(ft)  
  #html_grp = gdata.get_group(ft[0])  
  #plain_grp = gdata.get_group(ft[2])  
  cnt_stats = gdata['1'].count()
  cnt_stats.sort_values(inplace=True)
  
  # create filetype to index dictionary
  ft_to_idx = {}
  for idx, ft in enumerate(fts):
    ft_to_idx[ft] = idx

  nclasses = len(fts)

  return ft_to_idx, nclasses, grp


def subsampled_batch(ft_to_idx, group_data, class_size=100):
  
  
  fts = group_data.keys()
  subsampled = []
  
  for ft in fts:
    tmp = group_data[ft].sample(class_size, replace = True)
    subsampled = np.vstack((subsamlped, tmp.values))
  
  text_labels = subsamlped[:,0]
  idx_labels = map(lambda x: ft_to_idx[x], text_labels)
  x = subsamlped[:,1:]

  return x, idx_labels

def random_batch(data, batch_size, ft_to_idx, nclasses, onehot=False):
  # prepare random samples
  arr = np.arange(data.shape[0])
  np.random.shuffle(arr)
  raw = data.values
  
  # create training batch

  for i in range(raw.shape[0]/batch_size):
    batch = raw[i*batch_size : (i+1)*batch_size]

  batch_x = batch[:,1:]
  #TODO: add subsampling
  labels = batch[:,0]
  labels = map(lambda x: ft_to_idx[x], labels)
  if onehot == True:
    #convert to one-hot
    #may or may not use
    one_hot =  np.zeros((batch_size, nclasses))
    one_hot[np.arange(batch_size), labels] = 1.
    
    
    labels = one_hot

  return batch_x, labels

def mlp_model_fn(features, labels, mode, params):
  """Model function for MLP."""
  
  #TODO: need works on this
  config = params['config']


  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  
  input_layer = tf.reshape(features["x"], [-1, features["x"].shape[1], 1, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 256, 1, 1]
  # Output Tensor Shape: [batch_size, 252, 1, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 252, 1, 32]
  # Output Tensor Shape: [batch_size, 126, 1, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 126, 1, 32]
  # Output Tensor Shape: [batch_size, 122, 1, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 122, 1, 64]
  # Output Tensor Shape: [batch_size, 61, 1, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 61, 1, 64]
  # Output Tensor Shape: [batch_size, 61 * 1* 64]]
  pool2_flat = tf.reshape(pool2, [-1, 61 * 1 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 61 * 1* 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 93]
  logits = tf.layers.dense(inputs=dropout, units=93)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=93)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def my_input_fn(data_set):
  pass



def main(unused_argv):
  # Load training and eval data
  #TODO: figure out how unused_argv is for
  print ('unused_argv', unused_argv)

  ft_to_idx, nclasses, group_data = prepare_file(filename)
  train_data, train_labels = [], []
  for i in range(10):

    tmp_data, tmp_labels = subsampled_batch(ft_to_idx, group_data, class_size=100)
    train_data = np.vstack((train_data, tmp_data))
    train_labels = np.vstack((train_labels, tmp_labels))


  eval_data, eval_labels = subsampled_batch(ft_to_idx, group_data, class_size=100)
  #TODO: add a config with dict

  # Create the Estimator
  mlp_classifier = tf.estimator.Estimator(
    model_fn=mlp_model_fn, model_dir=config.path, params=config.dict)

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  
  mlp_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  
  eval_results = mlp_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
