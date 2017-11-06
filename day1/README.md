## Tensors
- 기본적인 arrays 

```
3
[1., 2.]
[[1., 2.], [3., 4.]]

```

- Tansor Rank : 몇차원 어레이냐

```
# 0 , Scalar
a = 123

# 1 , Vector
a = [1,2] 

# 2 , Matrix
a = [[1,2], [3,4]]

# 3 , 3-Tensor
# n , n-Tensor  
```

- Tensor Shapes : 각각의 element에 몇개씩 들어있느냐 
```
t=[[1,2,3], [4,5,6], [7,8,9]]
# [3,3] 3개씩 3개 들어있음 

```

- Tensor data type

```
tf.float32 # DT_FLOAT
tf.double # DT_DOUBLE
tf.int8 # DT_INT8
tf.int16
tf.int32
tf.int64
```


# 참고 강의
https://hunkim.github.io/ml/