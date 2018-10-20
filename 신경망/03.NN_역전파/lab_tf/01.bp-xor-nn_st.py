# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print('step_%d-> cost:%.3f' % (step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data})))
            print('step_%d->'%(step), 'W1:', sess.run(W1), 'W2:', sess.run(W2))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


#step_0-> cost:0.985
#step_0-> W1: [[-0.29956788 -1.6518148 ]
# [-0.592851   -1.4082304 ]] W2: [[ 0.6958311 ]
# [-0.47667417]]
#step_100-> cost:0.691
#step_100-> W1: [[-0.36264277 -1.6828035 ]
# [-0.6515919  -1.4525443 ]] W2: [[ 0.15527418]
# [-0.92668825]]
#...
#step_9900-> cost:0.014
#step_9900-> W1: [[-4.955657 -6.521944]
# [-4.956382 -6.526437]] W2: [[ 10.068679]
# [-10.57467 ]]
#step_10000-> cost:0.014
#step_10000-> W1: [[-4.9648952 -6.528671 ]
# [-4.9656157 -6.533132 ]] W2: [[ 10.094707]
# [-10.59925 ]]
#
#Hypothesis:  [[0.01099075]
# [0.98763263]
# [0.9876275 ]
# [0.01915845]] 
#Correct:  [[0.]
# [1.]
# [1.]
# [0.]] 
#Accuracy:  1.0

