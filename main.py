import numpy as np
import cupy as cp

x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)
print( l2_gpu )

x2 = x_gpu * x_gpu
print( x2 )


base = cp.array( [[1,2,3], [2,3,4], [3,4,5]] )
print( base )

x_by = x_gpu * base
print( cp.linalg.norm(x_by, axis=(0,)) )
