import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,1,1000)
y_gini = 1 - x**2 - (1-x)**2

y_mse = x**2

plt.plot(x,y_gini)
plt.plot(x, y_mse)
plt.show()