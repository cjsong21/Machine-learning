# -*- coding: utf-8 -*-
import tensorflow as tf

va = tf.Variable(5.0, name='va')
pa = tf.placeholder(tf.float32, name='pa')
print(pa)
# Tensor("pa:0", dtype=float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(va.eval(sess))
# 5.0
#print(pa.eval(sess))    #error

t = pa + 1.0
print(t.eval(session=sess, feed_dict={pa:7.3}))
# 8.3

print('-----------------')

ta = tf.placeholder(tf.float32, 3)
tb = tf.placeholder(tf.float32, 1)
tc = tf.multiply(ta, tb)
print(sess.run(tc, feed_dict={ta:[1.,2.,3.],tb:[3.]}))
# [3. 6. 9.]

print('-----------------')

sess.close()
