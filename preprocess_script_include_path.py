"""
Ming-Chang Chiu
Last Modified: 10/01/2017
Purpose: TO get filetype detected by Tika, file path and byte frequencies of TREC-DD Polar Dataset
Acknowledgement: convertToByteTable, compandBFD, and computeOnlyFingerPrint functions are 
modified from Rahul (https://github.com/USCDataScience/NN-fileTypeDetection)

"""

import numpy as np
import h5py
import os
from os import path
import preprocessor
import tika
from tika import detector
import sys
import pandas as pd

class preprocess():
  def __init__(self, beta=1.5):     
    self.path = os.getcwd()
    self.beta = beta
    self.dict = {}
    self.output = []

  def convertToByteTable(self, filename):
    """
    Converts the contents of the file to a 256 byte array
    input: filename
    output: byte table consisting of frequency distribution
    """
    try:
      table = [0] * 256
      #print filename
      data = open(filename, 'rb')
      #print 'ssssss open ok'
      buff = data.read(2 ** 20)
      while buff:
        for c in buff:
          table[ord(c)] += 1
        buff = data.read(2 ** 20)
      data.close()
      return table
    except:
      print 'Usage: %s <filename>' % os.path.basename(sys.argv[0])
      #self.logger('Usage: %s <filename>' % os.path.basename(sys.argv[0]))

  def compandBFD(self, table):
    """
    performs beta companding with beta value default as 1.5
    input: byte frequency table
    output: normalizes the values and compands to return a byte array.
    """
    table = [x * 1.0 / max(table) for x in table]
    table = [(x ** (1. / self.beta)) for x in table]
    return table
  def computeOnlyFingerPrint(self, filename):
    table  = self.convertToByteTable(filename)
    table = self.compandBFD(table)
    return table


def searchfile(path,postfix):
  for root, dirs, files in os.walk(path):
    for name in files:
      #print name       
      if name.endswith(postfix):
        #print root,name        
        return os.path.join(root,name) 
if __name__ == '__main__':
  

  pp = preprocess()  
  pp_nobeta =  preprocess(1)  
  temp_path = ''
  try:
    cnt = 0
    for root, dirs, files in os.walk(pp.path):

      for name in files:
        temp_path = os.path.join(root, name)
        if os.path.isfile(temp_path) and temp_path[-4:] != 'json' and temp_path[-2:] != 'py' and temp_path[-2:] != 'sh' and temp_path[-3:] != 'txt' and temp_path[-5:] != 'Store' and temp_path[-3:] != 'pyc' and temp_path[-3:] != 'csv':
          filetype = detector.from_file(temp_path) 
          #print filetype        
          table = pp.computeOnlyFingerPrint(temp_path)
          table.insert(0, filetype)
          table.insert(0, temp_path) # include file path to the dataset here
          pp.output.append(table)

          table_nobeta = pp_nobeta.computeOnlyFingerPrint(temp_path)
          table_nobeta.insert(0, filetype)
          table_nobeta.insert(0, temp_path) # include file path to the dataset here
          pp_nobeta.output.append(table_nobeta)

          #print pp.output
          #print 'qq'
          cnt+=1
        if cnt % 100 == 0:
          print cnt
        if cnt>0 and cnt % 10000 == 0:
          #print pp.output
          df = pd.DataFrame(pp.output)
          df.to_csv('temp_data.csv',sep=',', index=False)
          df = pd.DataFrame(pp_nobeta.output)
          df.to_csv('temp_nobeta_data.csv',sep=',', index=False)
          #print 'qq'
      for name in dirs: 
        temp_path = os.path.join(root, name)   
        if os.path.isfile(temp_path):                 
          filetype = detector.from_file(temp_path)         
          table = pp.computeOnlyFingerPrint(temp_path)
          table.insert(0, filetype)
          table.insert(0, temp_path) # include file path to the dataset here
          pp.output.append(table)

          table_nobeta = pp_nobeta.computeOnlyFingerPrint(temp_path)
          table_nobeta.insert(0, filetype)
          table_nobeta.insert(0, temp_path) # include file path to the dataset here
          pp_nobeta.output.append(table_nobeta)
          cnt+=1
          #print 'qq'
        if cnt % 100 == 0:
          print cnt
        if cnt>0 and cnt % 10000 == 0:
          #np.savetxt('temp_data.csv', np.asarray(pp.output), delimiter= ',', fmt = '%s')
          df = pd.DataFrame(pp.output)
          df.to_csv('temp_data.csv',sep=',', index=False)
          df = pd.DataFrame(pp_nobeta.output)
          df.to_csv('temp_nobeta_data.csv',sep=',', index=False)

  except:
    print 'FILE PATH: ', temp_path
    print 'NUM FILE: ', cnt
    df = pd.DataFrame(pp.output)
    df.to_csv('temp_data.csv',sep=',', index=False)
    df = pd.DataFrame(pp_nobeta.output)
    df.to_csv('temp_nobeta_data.csv',sep=',', index=False)

  df = pd.DataFrame(pp.output)
  df.to_csv('all_data.csv',sep=',', index=False) 
  df = pd.DataFrame(pp_nobeta.output)
  df.to_csv('all_nobeta_data.csv',sep=',', index=False)
  #np.savetxt('all_data.csv', np.asarray(pp.output), delimiter= ',')
