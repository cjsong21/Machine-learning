import tensorflow as tf
import numpy as np

data = [[2,0],[4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

# sigmoid function
y = 1/(1 + np.e**(a * x_data + b))

# -(y*logh + (1-y)*log(1-h))
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1-np.array(y_data)) * tf.log(1-y))

learning_rate = 0.5
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch_%5d -> COST=%.4f, 기울기(a=%.4f, 절편(b)=%.4f"
                  % (i, sess.run(loss), sess.run(a), sess.run(b)))
    
   
    sess.close()

#learning_rate = 0.5
#Epoch_    0 -> COST=1.2676, 기울기(a=0.1849, 절편(b)=-0.4334
#Epoch_ 6000 -> COST=0.0152, 기울기(a=-2.9211, 절편(b)=20.2982
#Epoch_12000 -> COST=0.0081, 기울기(a=-3.5637, 절편(b)=24.8010
#Epoch_18000 -> COST=0.0055, 기울기(a=-3.9557, 절편(b)=27.5463
#Epoch_24000 -> COST=0.0041, 기울기(a=-4.2380, 절편(b)=29.5231
#Epoch_30000 -> COST=0.0033, 기울기(a=-4.4586, 절편(b)=31.0675
#Epoch_36000 -> COST=0.0028, 기울기(a=-4.6396, 절편(b)=32.3346
#Epoch_42000 -> COST=0.0024, 기울기(a=-4.7930, 절편(b)=33.4086
#Epoch_48000 -> COST=0.0021, 기울기(a=-4.9261, 절편(b)=34.3406
#Epoch_54000 -> COST=0.0019, 기울기(a=-5.0436, 절편(b)=35.1636
#Epoch_60000 -> COST=0.0017, 기울기(a=-5.1489, 절편(b)=35.9005

