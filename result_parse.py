import json

index_result = json.loads( open('index_result.json').read() )

cat_words = {}
for index, result in index_result.items():
  #print( index, result) 
  cat = result['category']
  word = result['word']
  if cat_words.get( cat ) is None:
    cat_words[cat] = []
  cat_words[cat].append( word )

for cat, words in sorted( cat_words.items(), key=lambda x:x[0]):
  print( cat, sorted(words) )
