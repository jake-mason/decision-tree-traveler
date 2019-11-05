import numpy as np
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

bunch = load_boston()

X, y = bunch.data, bunch.target
feature_names = bunch.feature_names

model = DecisionTreeRegressor(
	max_depth=20,
	min_samples_leaf=2
)

model.fit(X, y)

class TreeInteractionFinder(object):

	def __init__(
		self,
		model,
		feature_names = None):

		self.model = model
		self.feature_names = feature_names

		self._parse_tree_structure()
		self._node_and_leaf_compute()

	def _parse_tree_structure(self):
		self.n_nodes = self.model.tree_.node_count
		self.children_left = self.model.tree_.children_left
		self.children_right = self.model.tree_.children_right
		self.feature = self.model.tree_.feature
		self.threshold = self.model.tree_.threshold
		self.n_node_samples = self.model.tree_.n_node_samples
		self.predicted_values = self.model.tree_.value

	def _node_and_leaf_compute(self):
		''' Compute node depth and whether each node is a leaf '''
		node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
		is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
		# Seed is the root node id and its parent depth
		stack = [(0, -1)]
		while stack:
			node_idx, parent_depth = stack.pop()
			node_depth[node_idx] = parent_depth + 1

			# If we have a test (where "test" means decision-test) node
			if self.children_left[node_idx] != self.children_right[node_idx]:
				stack.append((self.children_left[node_idx], parent_depth + 1))
				stack.append((self.children_right[node_idx], parent_depth + 1))
			else:
				is_leaves[node_idx] = True

		self.is_leaves = is_leaves
		self.node_depth = node_depth

	def node_and_leaf_print(self):
		for idx in range(self.n_nodes):
			node_prefix_fmter = '{}Node #{} (samples: {}'.format
			node_str = node_prefix_fmter(
				self.node_depth[idx] * '  ',
				idx,
				self.n_node_samples[idx]
			)
			dominant_prediction = self.predicted_values[idx].argmax()

			if self.is_leaves[idx]:
				node_str += '; final prediction: {:.2f})'.format(dominant_prediction)
				node_str += ' - leaf'
			else:
				node_str += '; dominant prediction: {:.2f}'.format(dominant_prediction)
				addl_metadata = ' - decision: go to node #{} if `{}` <= {:.2f} else to node #{}.'.format(
					self.children_left[idx],
					self.feature_names[self.feature[idx]],
					self.threshold[idx],
					self.children_right[idx]
				)
				node_str += addl_metadata
			print(node_str)

finder = TreeInteractionFinder(model, feature_names)
from collections import defaultdict

feature_combos = defaultdict(int)

for idx in range(finder.n_nodes):
	curr_node_is_leaf = finder.is_leaves[idx]
	curr_feature = finder.feature_names[finder.feature[idx]]
	if not curr_node_is_leaf:
		# Test to see if we're at the end of the tree
		try:
			next_idx = finder.feature[idx + 1]
		except IndexError:
			break
		else:
			next_node_is_leaf = finder.is_leaves[next_idx]
			if not next_node_is_leaf:
				next_feature = finder.feature_names[next_idx]
				feature_combos[frozenset({curr_feature, next_feature})] += 1

from pprint import pprint

pprint(sorted(feature_combos.items(), key=lambda x: -x[1])[:25])
pprint(sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]))
