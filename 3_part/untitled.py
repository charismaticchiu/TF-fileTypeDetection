import pandas as pd
import numpy as np


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
def subsampled_batch():
  pass


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
