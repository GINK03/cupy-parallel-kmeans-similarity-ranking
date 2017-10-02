import os
import sys
import math
import cupy as cp
import numpy as np
import pickle

index_meta = pickle.loads( open('make_word_vector/index_meta.pkl', 'rb').read() )

arrays = []
word_index = {}
index_word = {}
for index, meta in index_meta.items():
  #print( index )
  arrays.append( meta['vec'] )
  word = meta['word']
  word_index[word] = index
  index_word[index] = word

if '--cpu' in sys.argv:
  xp = np
if '--gpu' in sys.argv:
  xp = cp

x_all = xp.array(arrays) 
x_allnorm = xp.linalg.norm(x_all, axis=(1,) )

clusters =  [xp.random.randn(100) for n in range(100)] 
print( clusters )

rams = None
for it in range(100):
  cossimsall = []
  for e, cluster in enumerate(clusters):
    print( e )
    cluster_norm = xp.linalg.norm(cluster) 
    cluster_all = cluster * x_all
    norm = cluster_norm * x_allnorm
    invnorm = norm**-1 
    x_wa = (cluster * x_all).sum(axis=1) 
    cossims = x_wa * invnorm
    cossimsall.append( cossims )

  cossimsall = [ c.tolist() for c in cossimsall ]
  cc = np.array( cossimsall )
  print( cc.shape )
  print( cc.T )
  ams = np.argmax(cc.T, axis=1)
  if rams is not None and np.array_equal(rams,ams):
    print( ams )
    break
  rams = ams
  #print( ams )

  # 重心の計算
  am_vs = {}
  for e, am in enumerate(ams):
    #print(e, am) 
    if am_vs.get(am) is None:
      am_vs[am] = []
    am_vs[am].append( arrays[e] ) 

  means = []
  for am, vs in sorted(am_vs.items(), key=lambda x:x[0]):
    means.append( cp.mean(cp.array(vs), axis=0) )
  clusters = means

