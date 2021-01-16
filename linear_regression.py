# 모두를 위한 딥러닝 - TensorFlow로 간단한 Linear Regression을 구현 (new)

import tensorflow.compat.v1 as tf
tf.set_random_seed(777)  # for reproducibility

with tf.compat.v1.Session() as sess:
  x_train = [1,2,3]
  y_train = [1,2,3]

  W = tf.Variable(tf.random_normal([1]), name='weight')
  b = tf.Variable(tf.random_normal([1]), name='bias')

  hypothesis = x_train * W + b

  cost = tf.reduce_mean(tf.square(hypothesis - y_train))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  sess = tf.Session()
  #sess.run(tf.initialize_all_variables())
  sess.run(tf.global_variables_initializer())

  for step in range(2001):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))