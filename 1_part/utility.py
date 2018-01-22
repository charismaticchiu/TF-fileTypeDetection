from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle

def train_dev_split(group_data, proportion = 0.8, thre=10):
  """split all data into train and dev set"""
  train_dict = {}
  dev_dict = {}
  for k,v in group_data.iteritems():
    l = len(v)
    if l > thre:
      # prepare random samples
      arr = np.arange(l)
      np.random.shuffle(arr)
      train = v[arr[:int(proportion*l)]]
      dev = v[arr[int(proportion*l):]]
      train_dict[k] = train
      dev_dict[k] = dev
  return train_dict, dev_dict

def gen_synthetic(type, bf, num, sigma=10e-5):
  """generate synthetic data by adding small variance Gaussain noise
  bf: byte frequencies for the specific type
  num: number of synthetic sample points to be generated
  """
  
  def helper(x):
    if x > 0. and x < 1.:
      return x+np.random.gauss(0., sigma)
    else:
      return x

  synthetic = np.zeros(bf[0].shape)
  bf_mean = np.sum(bf,axis=0)/len(bf)
  # add noise
  for i in range(num):
    temp = np.array(map(lambda x:helper(x), bf_mean))
    synthetic = np.vstack((synthetic,temp))
  assert synthetic.shape[0] == num
  return synthetic

def prepare_file(filename):
  """Group data by filetype.
  ft_to_idx: filetype to index dict, e.g. ft_to_idx['html'] = 1
  nclasses: number of filetypes, should be 93
  grp: byte frequencies of each group, e.g., grp['html'] = a Dataframe that has 
       a list of byte frequencies of file belonging to that filetype
  """
  #filename = 'temp_data.csv'
  data = pd.read_csv(filename)
  fts = data['0'].unique()   
  gdata = data.groupby('0')  
  grp = {}
  for ft in fts:
    grp[ft] = gdata.get_group(ft).values  
  #html_grp = gdata.get_group(ft[0])  
  #plain_grp = gdata.get_group(ft[2])  
  #cnt_stats = gdata['1'].count()
  #cnt_stats.sort_values(inplace=True)
  
  # create filetype to index dictionary
  ft_to_idx = {}
  for idx, ft in enumerate(fts):
    ft_to_idx[ft] = idx

  nclasses = len(fts)
  #assert nclasses == 93
  print ('n classes: ',nclasses)

  return ft_to_idx, nclasses, grp



def subsampled_batch(ft_to_idx, group_data, class_size=30):
  """subsample each group to have 'class_size' number of samples
  and then convert their text labels to index labels.
  x: batch of bytefrequencies, with balanced sample size for each filetype
  idx_labels: index of the sample filetype

  the size of the batch will be class_size * nclasses
  """
  
  fts = group_data.keys()
  subsampled = np.zeros( (1, group_data['application/pdf'].shape[1]) )
  
  for ft in fts:
    #tmp = group_data[ft].sample(class_size, replace = True)
    tmp = group_data[ft][np.random.choice(group_data[ft].shape[0],class_size),:]
    #print tmp.shape
    subsampled = np.vstack((subsampled, tmp))
    #print subsampled.shape
  subsampled = np.delete(subsampled, 0, 0)
  text_labels = subsampled[:,0]
  idx_labels = np.array(map(lambda x: ft_to_idx[x], text_labels))
  #idx_labels = np.reshape(idx_labels,(1,-1))
  batch_x = subsampled[:,1:]
  assert idx_labels.shape[0] == batch_x.shape[0]
  return batch_x, idx_labels

def random_batch(data, batch_size, ft_to_idx, nclasses, onehot=False):
  """Get random batch of data.
  Can set labels to be one-hot or not
  data: should be a dictionary that has bytefrequencies as values
  """
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
  labels = np.array(map(lambda x: ft_to_idx[x], labels))
  #labels = np.reshape(labels,(1,-1))

  if onehot == True:
    #convert to one-hot
    #may or may not use
    one_hot =  np.zeros((batch_size, nclasses))
    one_hot[np.arange(batch_size), labels] = 1.
    
    
    labels = one_hot

  return batch_x, labels

def gen_feed(data, ft_to_idx, upper_limit=10000):
  """data: either data uesd to generate train or dev set
  upper_limit: upper bound number of sample points for each class 
  return: generated byte frequencies and labels seperately
  """
  keys = data.keys()
  generated = np.zeros((data[keys[0]].shape[1],))
  
  for k,v in data.iteritems():
    l = len(v)
    if l < upper_limit:
      generated = np.vstack((generated, data[k]))
    else:
      arr = np.arange(l)
      np.random.shuffle(arr)
      shuffled = v[arr[:upper_limit]]
      generated = np.vstack((generated, shuffled))
  generated = np.delete(generated,0,0)
  idx_labels = generated[:,0]
  idx_labels = np.array(map(lambda x: ft_to_idx[x], idx_labels))
  return generated[:,1:], idx_labels


