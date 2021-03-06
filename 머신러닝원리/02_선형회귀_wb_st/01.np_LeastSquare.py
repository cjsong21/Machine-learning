import numpy as np

x = [2,4,6,8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)
print('mean_x:%f, mean_y:%f' % (mx, my))
#mean_x:5.000000, mean_y:90.500000

divisor = np.sum([(i-mx)**2 for i in x])
print(divisor)
#20.0

def func_nu(x,mx,y,my):
    nu = 0
    for i in range(len(x)):
        nu += (x[i]-mx)*(y[i]-my)
    return nu

numerator = func_nu(x,mx,y,my)
print(numerator)
a = numerator / divisor
print(a)
b = my - (mx * a)
print(b)

#46.0
#2.3
#79.0