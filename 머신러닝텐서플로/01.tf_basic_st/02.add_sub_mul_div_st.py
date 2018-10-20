# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant(5, name='a')
b = tf.constant(2, name='b')

add1 = a + b
sub1 = a - b
mul1 = a * b
div1 = a / b
# 가독성 향상 코드
add2 = tf.add(a,b,name='add')
sub2 = tf.subtract(a, b, name='sub')
mul2 = tf.multiply(a,b, name='mul')
# div 결과는 정수
div2 = tf.div(a, b, name='div')
# divide 결과는 실수값
div3 = tf.divide(a, b, name='divide')

sess = tf.InteractiveSession()
print(add1.eval(), sub1.eval(), mul1.eval(), div1.eval())
print(add2.eval(), sub2.eval(), mul2.eval(), div2.eval(),div3.eval())

sess.close()

#7 3 10 2.5
#7 3 10 2 2.5