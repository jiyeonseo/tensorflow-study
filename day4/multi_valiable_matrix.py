# quiz1과 quiz2 그리고 mid-term 시험 점수로 final 점수 예측하기

import tensorflow as tf

# 쓰일 데아터

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# 플레이스홀더
X = tf.placeholder(tf.float32, shape=[None, 3]) # [ instance 갯수, variable 갯수 ] , None : 몇개가 들어올지 모르는 n
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Matrix를 사용함으로써 코드가 깔끔해짐.
# hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimize 시키기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# loop
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, "Cost :", cost_val, "\nPrediction:\n", hy_val)


