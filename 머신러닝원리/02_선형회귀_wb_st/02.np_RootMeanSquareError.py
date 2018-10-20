import numpy as np

#ab = [3,76]
#ab = [2,79]
ab = [2.3,79]
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]
#print(x)
#print(y)

# y = ax + b
def predict(x):
    return ab[0]*x + ab[1]

def rmse(p,y):
    return np.sqrt(((p-y)**2).mean())

def rmse_val(p_rst, y):
    return rmse(np.array(p_rst), np.array(y))

p_rst = []
for i in range(len(x)):
    p_rst.append(predict(x[i]))
    print('hour:%.1f, predit:%.1f' % (x[i], p_rst[i]))
    
print(rmse_val(p_rst, y))
    
#hour:2.0, predit:83.6
#hour:4.0, predit:88.2
#hour:6.0, predit:92.8
#hour:8.0, predit:97.4
#2.880972058177584