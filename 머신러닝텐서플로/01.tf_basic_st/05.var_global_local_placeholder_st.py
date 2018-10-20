# https://www.tensorflow.org/api_docs/python/tf/

import tensorflow as tf

vl = tf.local_variables()
vl = [[1,10],20,30]
print(vl)   
# [[1, 10], 20, 30]

vg = tf.Variable(tf.zeros(3, dtype=tf.int32), name='vg')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(vg))   
# [0 0 0]

td = tf.zeros((3,2))
print(sess.run(td))    
# [[0. 0.]
# [0. 0.]
# [0. 0.]]

ta = tf.placeholder(tf.float32, (2,2))
tb = tf.placeholder(tf.float32, (1,2))
tc = tf.multiply(ta, tb)
print(sess.run(tc, feed_dict={ ta:[[3.,2.],[2.,3.]], tb:[[4.,5.]] }))
# [[12. 10.]
# [ 8. 15.]]
sess.close()
