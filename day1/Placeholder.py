import tensorflow as tf

# 실행되는 시점에 변수로 넘기고 싶다.

a = tf.placeholder(tf.float32) # constant 대신 placeholder
b = tf.placeholder(tf.float32)

adder_node = a + b # tf.add(a, b)

sess = tf.Session()

# 실행 시점에 값을 넘겨준다
# sess.run(op, feed_dict={x: x_data})
print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # 7.5
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]})) # [ 3.  7.]


