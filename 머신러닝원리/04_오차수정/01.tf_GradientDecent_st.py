# Gradient Decent Example
import tensorflow as tf
tf.set_random_seed(666)

data = [[2,81],[4,93],[6,91],[8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

learning_rate = 0.1 # 0.01
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

y = a*x_data + b
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(4001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print("Epoch_%4d -> RMSE=%.4f, 기울기(a)=%.4f, 절편(b)=%.4f" 
                  % (step, sess.run(rmse), sess.run(a), sess.run(b)))

    sess.close()

#Epoch_   0 -> RMSE=69.8192, 기울기(a)=1.7147, 절편(b)=12.1788
#Epoch_ 100 -> RMSE=25.1977, 기울기(a)=12.5741, 절편(b)=17.6877
#Epoch_ 200 -> RMSE=23.5990, 기울기(a)=11.9134, 절편(b)=21.6312
#Epoch_ 300 -> RMSE=22.0035, 기울기(a)=11.2533, 절편(b)=25.5706
#Epoch_ 400 -> RMSE=20.4120, 기울기(a)=10.5940, 절편(b)=29.5051
#Epoch_ 500 -> RMSE=18.8257, 기울기(a)=9.9357, 절편(b)=33.4332
#Epoch_ 600 -> RMSE=17.2457, 기울기(a)=9.2788, 절편(b)=37.3535
#Epoch_ 700 -> RMSE=15.6741, 기울기(a)=8.6236, 절편(b)=41.2633
#Epoch_ 800 -> RMSE=14.1135, 기울기(a)=7.9707, 절편(b)=45.1595
#Epoch_ 900 -> RMSE=12.5680, 기울기(a)=7.3210, 절편(b)=49.0368
#Epoch_1000 -> RMSE=11.0435, 기울기(a)=6.6757, 절편(b)=52.8876
#Epoch_1100 -> RMSE=9.5497, 기울기(a)=6.0369, 절편(b)=56.6995
#Epoch_1200 -> RMSE=8.1023, 기울기(a)=5.4082, 절편(b)=60.4518
#Epoch_1300 -> RMSE=6.7281, 기울기(a)=4.7955, 절편(b)=64.1080
#Epoch_1400 -> RMSE=5.4733, 기울기(a)=4.2101, 절편(b)=67.6016
#Epoch_1500 -> RMSE=4.4131, 기울기(a)=3.6721, 절편(b)=70.8121
#Epoch_1600 -> RMSE=3.6401, 기울기(a)=3.2132, 절편(b)=73.5504
#Epoch_1700 -> RMSE=3.1932, 기울기(a)=2.8653, 절편(b)=75.6267
#Epoch_1800 -> RMSE=2.9933, 기울기(a)=2.6333, 절편(b)=77.0108
#Epoch_1900 -> RMSE=2.9187, 기울기(a)=2.4921, 절편(b)=77.8537
#Epoch_2000 -> RMSE=2.8934, 기울기(a)=2.4097, 절편(b)=78.3452
#Epoch_2100 -> RMSE=2.8850, 기울기(a)=2.3625, 절편(b)=78.6270
#Epoch_2200 -> RMSE=2.8823, 기울기(a)=2.3356, 절편(b)=78.7878
#Epoch_2300 -> RMSE=2.8814, 기울기(a)=2.3202, 절편(b)=78.8793
#Epoch_2400 -> RMSE=2.8811, 기울기(a)=2.3115, 절편(b)=78.9313
#Epoch_2500 -> RMSE=2.8810, 기울기(a)=2.3065, 절편(b)=78.9610
#Epoch_2600 -> RMSE=2.8810, 기울기(a)=2.3037, 절편(b)=78.9778
#Epoch_2700 -> RMSE=2.8810, 기울기(a)=2.3021, 절편(b)=78.9874
#Epoch_2800 -> RMSE=2.8810, 기울기(a)=2.3012, 절편(b)=78.9928
#Epoch_2900 -> RMSE=2.8810, 기울기(a)=2.3007, 절편(b)=78.9959
#Epoch_3000 -> RMSE=2.8810, 기울기(a)=2.3004, 절편(b)=78.9977
#Epoch_3100 -> RMSE=2.8810, 기울기(a)=2.3002, 절편(b)=78.9987
#Epoch_3200 -> RMSE=2.8810, 기울기(a)=2.3001, 절편(b)=78.9992
#Epoch_3300 -> RMSE=2.8810, 기울기(a)=2.3001, 절편(b)=78.9996
#Epoch_3400 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=78.9998
#Epoch_3500 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=78.9999
#Epoch_3600 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=78.9999
#Epoch_3700 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=79.0000
#Epoch_3800 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=79.0000
#Epoch_3900 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=79.0000
#Epoch_4000 -> RMSE=2.8810, 기울기(a)=2.3000, 절편(b)=79.0000
