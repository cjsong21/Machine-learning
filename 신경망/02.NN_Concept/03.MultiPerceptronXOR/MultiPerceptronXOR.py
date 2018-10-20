# XOR 문제해결 using Multilayer Perceptron
import numpy as np

w11 = np.array([-2, -2])   #w11 = np.array([-1, -1])  --> NOK
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1            #b2 = 1  --> NOK
b3 = -1

# mutilayer perceptron function
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0 :
        return 0
    else:
        return 1

# NAND Gate
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

# OR Gate
def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

# AND Gate
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)

# XOR Gate
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

# output print
if __name__ == '__main__':
    for x in [(0,0), (1,0), (0,1), (1,1)]:
        y = OR(x[0], x[1])
        print("or_input : " + str(x) + "  --> or_output : " + str(y))

    print('--------------------------------------')
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(x[0], x[1])
        print("and_input : " + str(x) + "  --> and_output : " + str(y))

    print('--------------------------------------')
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(x[0], x[1])
        print("xor_input : " + str(x) + "  --> xor_output : " + str(y))


#or_input : (0, 0)  --> or_output : 0
#or_input : (1, 0)  --> or_output : 1
#or_input : (0, 1)  --> or_output : 1
#or_input : (1, 1)  --> or_output : 1
#--------------------------------------
#and_input : (0, 0)  --> and_output : 0
#and_input : (1, 0)  --> and_output : 0
#and_input : (0, 1)  --> and_output : 0
#and_input : (1, 1)  --> and_output : 1
#--------------------------------------
#xor_input : (0, 0)  --> xor_output : 0
#xor_input : (1, 0)  --> xor_output : 1
#xor_input : (0, 1)  --> xor_output : 1
#xor_input : (1, 1)  --> xor_output : 0

