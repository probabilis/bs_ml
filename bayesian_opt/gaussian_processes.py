import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from repo_utils import fontsize, fontsize_title

x = np.linspace(-1,1,1000)

def gaussian(x, theta_d):
    r = x**2 / theta_d
    K = theta_d * np.exp(-0.5 * r)
    return K

def matern(x,theta_d):
    r = x**2 / theta_d
    f = (5* r**2)**(0.5)
    K = theta_d * (1 + f + 5/3 * r) * np.exp( -(f) )
    return K

theta_d = 0.1
y = gaussian(x,theta_d)
z = matern(x,theta_d)

fig = plt.gcf()
fig.set_size_inches(6,6)
plt.plot(x,y, color = "salmon", label = "Gaussian")
plt.plot(x,z, color = "cornflowerblue", label = "Matern")
plt.xlabel("$x$", fontsize = fontsize)
plt.ylabel("$f(x)$", fontsize = fontsize)
plt.title(f"Comparison of pure Gaussian with \n Matern covariance function $\\nu = $ {theta_d}", fontsize = fontsize_title)
plt.legend()
fig.tight_layout()
plt.show()
