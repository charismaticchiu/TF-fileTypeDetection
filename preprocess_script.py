import numpy as np
import h5py
import os
from os import path
import preprocessor
import tika
from tika import detector
import sys

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

  for root, dirs, files in os.walk(pp.path):

    for name in files:
      temp_path = os.path.join(root, name)
      if os.path.isfile(temp_path) and temp_path[-4:] != 'json' and temp_path[-2:] != 'py' and temp_path[-2:] != 'sh' and temp_path[-3:] != 'txt' and temp_path[-5:] != 'Store' and temp_path[-3:] != 'pyc':
        filetype = detector.from_file(temp_path) 
        print filetype        
        table = pp.computeOnlyFingerPrint(temp_path)
        pp.output.append([filetype, table])
        

    for name in dirs: 
      temp_path = os.path.join(root, name)   
      if os.path.isfile(temp_path):                 
        filetype = detector.from_file(temp_path)         
        table = pp.computeOnlyFingerPrint(temp_path)
        pp.output.append([filetype, table])

  np.savetxt('all_data.csv', np.asarray(pp.output), delimiter= ',')
