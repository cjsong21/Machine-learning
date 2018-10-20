# Linear Regression (w=2)

import tensorflow as tf
# input
x = [1., 2., 3., 4.]
# label
y = [2., 4., 6., 8.]
m = n_samples = len(x)

w = tf.placeholder(tf.float32)
hypo = tf.multiply(x, w)        # H(x) = wx
cost = tf.reduce_sum(tf.pow(hypo - y, 2))/(m)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init);

for i in range(-20, 30):
    print("i:%d, w:%.1f, cost:%f" % (i, i*0.1, sess.run(cost, feed_dict={w: i*0.1})))

#i:-20, w:-2.0, cost:120.000000
#i:-19, w:-1.9, cost:114.075005
#i:-18, w:-1.8, cost:108.299995
#i:-17, w:-1.7, cost:102.675003
#i:-16, w:-1.6, cost:97.199997
#i:-15, w:-1.5, cost:91.875000
#i:-14, w:-1.4, cost:86.699997
#i:-13, w:-1.3, cost:81.674995
#i:-12, w:-1.2, cost:76.800003
#i:-11, w:-1.1, cost:72.074997
#i:-10, w:-1.0, cost:67.500000
#i:-9, w:-0.9, cost:63.075005
#i:-8, w:-0.8, cost:58.799995
#i:-7, w:-0.7, cost:54.675003
#i:-6, w:-0.6, cost:50.699997
#i:-5, w:-0.5, cost:46.875000
#i:-4, w:-0.4, cost:43.200001
#i:-3, w:-0.3, cost:39.674999
#i:-2, w:-0.2, cost:36.299999
#i:-1, w:-0.1, cost:33.074997
#i:0, w:0.0, cost:30.000000
#i:1, w:0.1, cost:27.074999
#i:2, w:0.2, cost:24.299999
#i:3, w:0.3, cost:21.674999
#i:4, w:0.4, cost:19.200001
#i:5, w:0.5, cost:16.875000
#i:6, w:0.6, cost:14.699999
#i:7, w:0.7, cost:12.674999
#i:8, w:0.8, cost:10.800000
#i:9, w:0.9, cost:9.075001
#i:10, w:1.0, cost:7.500000
#i:11, w:1.1, cost:6.074999
#i:12, w:1.2, cost:4.799999
#i:13, w:1.3, cost:3.675001
#i:14, w:1.4, cost:2.700000
#i:15, w:1.5, cost:1.875000
#i:16, w:1.6, cost:1.200000
#i:17, w:1.7, cost:0.675000
#i:18, w:1.8, cost:0.300000
#i:19, w:1.9, cost:0.075000
#i:20, w:2.0, cost:0.000000
#i:21, w:2.1, cost:0.075000
#i:22, w:2.2, cost:0.300000
#i:23, w:2.3, cost:0.675000
#i:24, w:2.4, cost:1.200001
#i:25, w:2.5, cost:1.875000
#i:26, w:2.6, cost:2.699999
#i:27, w:2.7, cost:3.675001
#i:28, w:2.8, cost:4.799999
#i:29, w:2.9, cost:6.075002
    