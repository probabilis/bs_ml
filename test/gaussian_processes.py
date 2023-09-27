import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,1000)

def gaussian(x, theta_d):
    r = x**2 / theta_d
    K = theta_d * np.exp(-0.5 * r)
    return K

theta_d = 1

y = gaussian(x,theta_d)

def matern(x,theta_d):
    r = x**2 / theta_d
    f = (5* r**2)**(0.5)
    K = theta_d * (1 + f + 5/3 * r) * np.exp( -(f) )
    return K

z = matern(x,theta_d)

plt.plot(x,y)
plt.plot(x,z)
plt.show()
