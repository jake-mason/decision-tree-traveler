'''
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from dtree_interaction import TreeInteractionFinder

bunch = load_boston()

X, y = bunch.data, bunch.target
feature_names = bunch.feature_names

model = DecisionTreeRegressor(
	max_depth=20,
	min_samples_leaf=2
)

model.fit(X, y)

finder = TreeInteractionFinder(model, feature_names)
'''

from collections import defaultdict

import numpy as np

class TreeInteractionFinder(object):

	def __init__(
		self,
		model,
		feature_names = None):

		self.model = model

		if isinstance(feature_names, np.ndarray):
			criterion = feature_names.size == 0
		else:
			criterion = not feature_names

		if criterion:
			feature_names = list(range(model.n_features_))

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

	def find_interactions(self):
		'''
		Crawl tree and find potential interactions by
		looking at which features tend to follow particular
		features as the tree is traversed

		Note: this only works for single-tree classifiers (i.e.
		this has only been tested on
		sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor)
		'''

		feature_combos = defaultdict(int)

		for idx in range(self.n_nodes):
			curr_node_is_leaf = self.is_leaves[idx]
			curr_feature = self.feature_names[self.feature[idx]]
			if not curr_node_is_leaf:
				# Test to see if we're at the end of the tree
				try:
					next_idx = self.feature[idx + 1]
				except IndexError:
					break
				else:
					next_node_is_leaf = self.is_leaves[next_idx]
					if not next_node_is_leaf:
						next_feature = self.feature_names[next_idx]
						feature_combos[frozenset({curr_feature, next_feature})] += 1

		self.feature_combos = feature_combos
		return feature_combos