import pickle
import numpy as np
import cupy as cp
import time
import sys

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

bench = False
if '--bench' in sys.argv:
  bench = True 

x_all = xp.array(arrays) 
x_allnorm = xp.linalg.norm(x_all, axis=(1,) )

start = time.time()
for e, (word, index) in enumerate(word_index.items()):
  print( word )
  x_word = xp.array( arrays[index] )
  x_wa = (x_word * x_all).sum(axis=1)
  x_wordnorm = xp.linalg.norm(x_word)
  norm = x_wordnorm * x_allnorm
  invnorm = norm**-1

  cossims = x_wa * invnorm
  weight_term = [(w, index_word[i]) for i, w in enumerate(cossims.tolist())]
  topn = sorted( weight_term, key=lambda x:x[0]*-1)[:31]
  print(topn)
  if e > 10 and bench:
    break

print('elapsed', time.time() - start )
