import tensorflow as tf

hello = 'Hello, TensorFlow!'
print(hello)

hello = tf.constant('Hello, TensorFlow!!')
print(hello)

sess = tf.Session()
sess.run(hello)
print(hello.eval(session=sess))

sess.close()


#
#
#data = [[2,81],[4,93],[6,91],[8,97]]
#x_data = [x_row[0] for x_row in data]
#y_data = [y_row[1] for y_row in data]
#learning_rate = 0.1
