import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

# 위 train 값을 직접 주는게 아니라 placeholder를 통해 run 하는 시점에 넘길 수 있다
X = tf.placeholder(tf.float32, shape=[None]) # shape=[None] 학습데이터를 무제한으로 넣을 수 있다.
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name="weight") # Variable : trainable한 값
b = tf.Variable(tf.random_normal([1]), name="bias") # 아직 shape이 어케 생겼는지 모르니까 random_normal 으로 준다

# Hypothesis = W(x) + b
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y)) # cost/lost function

# Minimize
## GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 이 cost 를 minimize 해라

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # W, b 는 tf의 variable. 쓰기전에 반드시 initialize를 해주어야 한다


# fit the line
# for step in range(2001) :
#     sess.run(train)
#     if step % 20 == 0 :
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3,4,5], Y:[1.1,2.1,3.1,4.1,5.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# 결과
# 처음에는 cost 가 컸다가 점점 작아짐
# W 는 점점 1로 b는 0으로 수렴

# placeholder를 쓰면 linear hypothesis를 먼저 만들어놓고 학습 데이터를 나중에 넣을 수 있는 장점

# test
print(sess.run(hypothesis, feed_dict={X : [5]}))
print(sess.run(hypothesis, feed_dict={X : [2.5]}))