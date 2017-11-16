## recap

### Hypothesis
- H(x) = Wx + b

### Cost Function(Loss)

### Gradient Descent algorithm

# multi-variable linear regression 

- 여러개의 인풋
- quiz1과 quiz2 그리고 mid-term 시험 점수로 final 점수 예측하기
- 변수가 3개 
 
### Hypothesis
- H(x) = Wx + b # 변수가 하나일때 
- H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b # 변수가 하나일때 
- 변수가 많아질수록 w1x1 + w2x2 + w3x3 ... + wnxn    
- Matrix를 이용하자 

## Matrix multiplication
- [5, 3] * [3,1] = [5,1]
- [instance(데이터 샘플) 갯수 * 변수(feature) 갯수] * [weight 갯수 , 1 ]
- [a,b] * [b,c] = [a,c] 이렇게 나옴

# CSV 데이터 읽어오기 
- numpy에 있는 loadtxt
```
import numpy as np

xy = np.loadtxt('aa.csv', delimiter=',', dtype=np.float32)
# 파일이름, 자르는 기준값, 데이터타입 
```

# Queue runner 
- 파일의 크기가 너무 커서 numpy로 메모리에 올리기 부담스러울때!
