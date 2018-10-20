# -*- coding: utf-8 -*-
import tensorflow as tf

hello = 'Hello, TensorFlow!'
print(hello)
#Hello, TensorFlow!

hello = tf.constant('Hello, TensorFlow!!')
print(hello)
#Tensor("Const_3:0", shape=(), dtype=string)

sess = tf.Session()
sess.run(hello)
print(hello.eval(session=sess))
#b'Hello, TensorFlow!!'

sess.close()
