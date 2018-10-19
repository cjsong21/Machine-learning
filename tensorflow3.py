import tensorflow as tf

a = tf.constant(5,name='a')
b = tf.constant(2,name='b')

va = tf.Variable(5, name='va')
vb = tf.Variable(3, name='vb')
vc = tf.Variable(tf.zeros(0,tf.int32), name='vc')

sess = tf.Session()

print(sess.run(a))

sess.run(tf.global_variables_initializer())
print(sess.run(va))
print(va.eval(sess))

va = va + 10
vb = vb - 5
vc = va
vc = vc + vb

print(sess.run(va))
print(sess.run(vb))
print(sess.run(vc))

sess.close()
