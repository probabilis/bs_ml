import sklearn
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from decision_regression_tree_from_scratch import DecisionTreeRegressorScratch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################

max_depth = 3

x_min = 0
x_max = 2 * np.pi
N = 100

X = np.linspace(x_min,x_max,100)
Y = np.sin(X)
X = X.reshape(-1,1)

model = DecisionTreeRegressor(max_depth = max_depth)
model.fit(X,Y)
y_pred = model.predict(X)

model_scratch = DecisionTreeRegressorScratch(pd.DataFrame(X), Y.tolist(), max_depth = max_depth)
model_scratch.grow_tree()
y_pred_scratch = model_scratch.fit(pd.DataFrame(X))

################################################

print("hola")

fig, ax = plt.subplots(1)
fig.set_size_inches(12,8)

fig.suptitle("Decision Tree Regressor from Scratch comparison with SKLEARN module")
ax.set_title("max depth $d_{max}$ = " + str(max_depth))
ax.scatter(X, Y, color = 'gray')
ax.plot(X, y_pred, color = 'salmon', label = 'SKLEARN')
ax.plot(X, y_pred_scratch, color = 'mediumseagreen', label = 'SCRATCH')
ax.legend()
fig.tight_layout()
plt.savefig("decision_tree_regressor_comparison.png")
plt.show()

from sklearn import tree


#import graphviz
#dot_data = export_graphviz(model, out_file = None)
#graph = graphviz.Source(dot_data, filename = 'test.gv', format = 'png')
#graph.view()


tree.plot_tree(model)
fig = plt.gcf()
#fig.set_size_inches(16,14)
fig.tight_layout()
plt.savefig("decision_tree_plot.png")
plt.show()