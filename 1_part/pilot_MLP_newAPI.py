from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
from utility import *



tf.logging.set_verbosity(tf.logging.INFO)


def mlp_model_fn(features, labels, mode, params):
  """Model function for MLP."""
  
  #TODO: need works on this
  config = params


  # Input Layer
  
  input_layer = tf.reshape( features["x"], [-1, features["x"].shape[1] ] )
  #print ('feature x', features["x"])
  #print ('feature x shape', features["x"].shape)
  #print ('reshape:', input_layer)
  #print ('reshape shape:', input_layer.shape)
  #trans = tf.string_to_number(input_layer)
  #print ('trans reshape:', trans)
  #print ('reshape shape:', input_layer.shape)



  # Dense Layers

  hidden1 = tf.layers.dense(inputs=features["x"], units=config['n_hidden1'], activation=tf.nn.relu)
  drop_h1 = tf.layers.dropout(
      inputs=hidden1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  hidden2 = tf.layers.dense(inputs=drop_h1, units=config['n_hidden2'], activation=tf.nn.relu)
  drop_h2 = tf.layers.dropout(
      inputs=hidden2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  hidden3 = tf.layers.dense(inputs=drop_h2, units=config['n_hidden3'], activation=tf.nn.relu)
  drop_h3 = tf.layers.dropout(
      inputs=hidden3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=drop_h3, units=config['nclasses'])
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
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config['nclasses'])
  print (onehot_labels)
  print (logits)
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
  #unused_argv: a list of strings
  print ('unused_argv', unused_argv)
  filename = unused_argv[1]


  if os.path.isfile('ft_to_idx.npy') and os.path.isfile('nclasses.npy') and os.path.isfile('group_data.npy'):
    ft_to_idx = np.load('ft_to_idx.npy')
    ft_to_idx = ft_to_idx.item()
    nclasses = np.load('nclasses.npy')
    #f = open('group_data.pkl','r')
    #group_data = pickle.load('group_data.pkl')
    group_data = np.load('group_data.npy')
    group_data = group_data.item()

  else:
    ft_to_idx, nclasses, group_data = prepare_file(filename)
    np.save("ft_to_idx", ft_to_idx)
    np.save("nclasses", nclasses)
    #f = open('group_data.pkl','w')
    #pickle.dump(group_data, f)
    np.save("group_data", group_data)

  #gen train set
  train, dev = train_dev_split(group_data, proportion = 0.8, thre=10)
  train_data, train_labels= gen_feed(train, ft_to_idx, upper_limit=5000)
  eval_data, eval_labels= gen_feed(dev, ft_to_idx, upper_limit=5000)
  train_data = train_data.astype(np.float32) 
  eval_data = eval_data.astype(np.float32) 
  '''
  train_data, train_labels = np.zeros((1, group_data['application/pdf'].shape[1]-1)), np.zeros((1,))
  for i in range(100):
    tmp_data, tmp_labels = subsampled_batch(ft_to_idx, group_data, class_size=100)
    train_data = np.vstack((train_data, tmp_data))
    train_labels = np.hstack((train_labels, tmp_labels))
  train_data = np.delete(train_data,0,0)
  train_labels = np.delete(train_labels,0,0)
  train_data = train_data.astype(np.float32)
  '''
  #gen dev set

  '''eval_data, eval_labels = subsampled_batch(ft_to_idx, group_data, class_size=100)
  eval_data = eval_data.astype(np.float32) 
  '''
  # set config
  config = {}
  config['nclasses'] = int(nclasses)
  config['model_dir'] = unused_argv[2]
  config['n_hidden1'] = int(unused_argv[3])
  config['n_hidden2'] = int(unused_argv[4])
  config['n_hidden3'] = int(unused_argv[5])
  

  # Create the Estimator
  mlp_classifier = tf.estimator.Estimator(
    model_fn=mlp_model_fn, model_dir=config['model_dir'], params=config)

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
      steps=60000,
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
