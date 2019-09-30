# decision-tree-traveler

An expansion upon [this](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-download-auto-examples-tree-plot-unveil-tree-structure-py) scikit-learn tutorial.

```python
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from tree_explainer import DecisionTreeExplainer

bunch = load_boston()

X, y = bunch.data, bunch.target
feature_names = bunch.feature_names

model = DecisionTreeRegressor(
	max_depth=5,
	min_samples_leaf=10
)

model.fit(X, y)
ypred = model.predict(X)

expl = DecisionTreeExplainer(
	X,
	model,
	y,
	ypred,
	feature_names
)

expl.calculate_decision_paths(X)
print(expl.decision_path(0))
# RM (value = 6.575) less than or equal to 6.94
#   and LSTAT (value = 4.98) less than or equal to 14.40
#    and DIS (value = 4.09) greater than 1.38
#     and RM (value = 6.575) greater than 6.54
#      and TAX (value = 296.0) greater than 269.00
#       and NOX (value = 0.538) greater than 0.53
#        * Final prediction: 23.47
```