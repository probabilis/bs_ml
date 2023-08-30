import numpy as np

###################################

x_min = 0.1
x_max = 10
N = 200

X = np.linspace(x_min ,x_max , N)

def testfunction(x, noise):
    return (np.log(x) + np.sin(x)) + noise * np.random.randn(*x.shape)