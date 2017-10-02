# GPU Accelerated K-means

## 既存のkmeansは遅すぎる
CPUでやった場合のkmeansはscikit-learnなどを使えるのですが、いかんせん遅すぎるのと、kmeansの多くは距離の定義がEuclidであって、word2vecなどのベクトル情報をクラスタリングする際には、あまり速度が出ません  
また、実際に、kmeansの学習ステップを見るとほとんどのクラスが安定しているのに、更新が止まるまで繰り返してしまうので、かなり遅いです。　　

十分な精度やイテレーションを繰り返した上で、停止できるように設計します  

## Cupy
Cupyはchainerのバックエンドで用いられているnumpyの一部語感ライブラリであり、cupyはこの行列の計算をGPUで行うことができます  

numpyの機能で高機能のものは、まだ使えない面も多いですが、最初にnumpy, cupy双方コンパチブルに動作させるようなことができる場合、CPUで計算させたほうがいい場合、
GPUで計算させたほうがいい場合、双方の最適なコードを記述することができます　

四則演算と、ノルムと、dot積などよく使う系のオペレーションを10回, 10000x10000の行列で計算するベンチマークを入れてみました  
```python
import sys
import time
import numpy as np
import cupy as cp

xp = np
if '--gpu' in sys.argv:
  xp = cp
  
b = xp.random.randn(10000, 10000)

start = time.time()
for i in range(10):
  m = b*b
  d = b - b
  a = b + b
  de = b/b
  xp.dot(b, b)
  xp.linalg.norm(b)
  inv = b**-1
  print( 'now iter', i )

print('elapsed', time.time() - start )
```

### CPU(Ryzen 1700X)で実行
numpyもマルチプロセスで動作するので、Ryzenのような多コアで計算すると、高速に計算することが期待できます　　

htopでCPUの使用率を見ると、16スレッド全て使い切っています.numpy優秀ですね  

<p align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/31068076-44e3e8d6-a791-11e7-86f7-52beded4b48e.png">
</p>
<div align="center"> 図.1 numpyのCPUの使用率 </div>

```cosnole
$ python3 bench.py --cpu
now iter 0
now iter 1
now iter 2
now iter 3
now iter 4
now iter 5
now iter 6
now iter 7
now iter 8
now iter 9
elapsed 304.03318667411804 #<- 5分かかっている
```

### GPU(Nvidia GeForce GTX1700)で実行
cupyで実行します  
cupyで実行する際には、Ryzenのコアは1個に限定される様子が観測できます  
<p align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/31068350-928aad4e-a792-11e7-9ef9-6027b0059237.png">
</p>
<div align="center"> 図2. cupyのCPUの使用率 </div>

その代わり、GPUはフル回転している様子が観測できます  
<p align="center">
  <img width="600px" src="https://user-images.githubusercontent.com/4949982/31068402-ce71b94c-a792-11e7-8173-1c7a335bf137.png">
</p>
<div align="center"> 図3. CPUの使用率の様子 </div>
```console
$ python3 bench.py --gpu                                                                                          
now iter 0                                      
now iter 1                                      
now iter 2                                      
now iter 3
now iter 4
now iter 5
now iter 6
now iter 7
now iter 8
now iter 9
elapsed 102.1835367679596 # <- 100秒程度に短縮できた
```
Ryzenと比較しても3倍程度早くなることがわかりました  
CPUはSingle Threadで動作するので、他の分析オペレーションを回すことができます  

