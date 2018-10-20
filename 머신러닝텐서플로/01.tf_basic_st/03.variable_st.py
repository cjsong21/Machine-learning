# -*- coding: utf-8 -*-
import tensorflow as tf
# 지역변수 정의
a = tf.constant(5,name='a')
b = tf.constant(2,name='b')
# 전역변수 선언
va = tf.Variable(5, name='va')
vb = tf.Variable(3, name='vb')
vc = tf.Variable(tf.zeros(0,tf.int32), name='vc')

sess = tf.Session()

print(sess.run(a))  #5
# 전역변수는 초기값 설정이후 사용가능함
sess.run(tf.global_variables_initializer())
print(sess.run(va))  #5
print(va.eval(sess))  #5

va = va + 10
vb = vb - 5
vc = va
vc = vc + vb

print(sess.run(va))
print(sess.run(vb))
print(sess.run(vc))

sess.close()

#15
#-2
#13

