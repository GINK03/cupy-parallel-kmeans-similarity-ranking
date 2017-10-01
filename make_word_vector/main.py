import gzip
import re
import MeCab
import sys

if '--wakati' in sys.argv:
  m = MeCab.Tagger('-Owakati')
  with gzip.open('../../../yahoo.tar.gz', 'rt') as f:
    for line in f:
      if line.strip() == '': continue
      text = re.sub(r'\./.*$', '', line.strip()).strip()
      words = m.parse(text).strip()
      print( words )

import pickle
if '--make_dataframe' in sys.argv:
  index_meta = {}
  with open('model.vec') as f:
    for line in f:
      line = line.strip()
      ents = line.split()
      word = ents.pop(0)
      try:
        vec = [float(v) for v in ents]
      except ValueError as e:
        continue
      if len(vec) != 100:
        continue
      print( word )
      index_meta[len(index_meta)] = {'word':word, 'vec':vec}
  open('index_meta.pkl', 'wb').write( pickle.dumps(index_meta) )


