 ## Regression
 - supervised learning
 - 얼마나 공부했더니 얼마나 점수가 나오더라
 - 범위가 있는 것 (0점~100점)
 - 어떠한 데이터 -> training -> regression model
 
### (Linear) Hypothesis 
 - ex) 
    - 훈련을 많이 할수록 잘한다
    - 집 크기가 클수록 비싸다
 - Linear 한 선을 찾는다 == 학습한다 
 - H(x) = W(x) + b  
  

### Cost Function 
 - Linear 선에서 얼마나 먼가 (거리를 측정)
 - Lost Function 
 - H(x) - y // 가설 - 실제값
 - ( H(x) - y )^2 // 
 - 1/m ( H(x) - y )^2 //
 - Goal >> minimize cost = cost (W, b) // 평균값!
 - 가장 작은 값을 찾는것 == 학습한다.
  