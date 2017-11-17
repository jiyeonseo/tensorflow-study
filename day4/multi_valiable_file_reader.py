# 파일을 읽어와서 Queue runner를 이용해 데이터를 넣어본다.

import tensorflow as tf

# ==================== 변경부분
# 쓰일 데아터
file_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name="filename_queue")

# 데이터 읽기
reader = tf.TextLineReader()
key, value = reader.read(file_queue)

# 데이터 읽어 온 값을 변수에
record_defaults = [[0.],[0.],[0.],[0.]] # 이러한 형태로 들어올 것이다. ( & 디폴트 값 설정 )
xy = tf.decode_csv(value, record_defaults=record_defaults)


# 배치로 가져오도록
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # x값, y값, batch는 몇개씩 가져올 것인가
# ==================== 변경부분

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

# ==================== 변경부분
# 파일에서 꺼내오는 복잡한 부분은 텐서플로가 알아서
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)


# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))


# ==================== 변경부분