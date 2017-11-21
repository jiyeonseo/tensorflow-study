# Logistic Classification
###  linear regression과 다른점
- linear regression은 어떠한 숫자를 예측 
- classification 어느것을 고른다.  

## Binary Classification
- 둘 중 하나를 고른다.
- 스팸이냐(1) 아니냐(0) 
- 페이스북에서 포스트를 보일꺼냐(1) 말꺼냐(0) 
- 주식을 살까(1) 팔까(0)
- 시험을 통과할까(1) 실패할까(0)

- linear regression을 쓰면 간단하고 좋긴 한데 classification에서 필요한 0과 1 이외의 너무 큰 수들이 나타난다 
- ex) w=0.5, b=0 인 hypothesis에서 100시간을 공부한 사람 = 50 
    - 우리가 알고자 하는 값(1 또는 0)과 너무 다르다.
- g(z) = 0과 1사이의 값 
    - sigmoid : S 자 처럼 그래프가 나와서 
    - logistic function, sigmoid function 
      
## Logistic function
- 가로가 z축 / 세로가 g(z) 축
- z가 커질수록 g(z) 값은 1에 가까워짐 
- z가 작아지면 g(z) 값은 0에 가까워짐
- 0보다 작아지거나 1보다 커지지 않음.


### cost function 
- U자형 그래프에서 가장 낮은 지점 = cost가 가장낮은 지점 = global minimum
 
### gradient decent
```buildoutcfg
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
                       
# Minimize
a = tf.Variable(0.1) # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

```   

### training data
- 결과값(y_data)은 항상 0 또는 1이 되어야 한다

## 참고
- 데이터 kaggle (https://www.kaggle.com/)   