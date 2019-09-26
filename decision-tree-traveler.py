'''
https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-download-auto-examples-tree-plot-unveil-tree-structure-py

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston

from tree_explainer import DecisionTreeExplainer

bunch = load_boston()

X, y = bunch.data, bunch.target
feature_names = bunch.feature_names

model = DecisionTreeRegressor(
	max_depth=5,
	min_samples_leaf=10
)

model.fit(X, y)

try:
	event_mapping = {idx: str(name) for idx, name in enumerate(bunch.target_names)}
except AttributeError:
	event_mapping = None

ypred = model.predict(X)

expl = DecisionTreeExplainer(
	X,
	model,
	y,
	ypred,
	feature_names,
	event_mapping
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
'''

import warnings
import subprocess
from typing import List, Dict, NoReturn, Tuple, Any, Union, Iterable

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

READABLE_SIGN_MAPPING = {'<=': 'less than or equal to', '>': 'greater than'}

def ndarray_exists(array: np.ndarray) -> bool:
	return array is not None and len(array) > 0

class DecisionTreeExplainer(object):

	def __init__(
		self,
		X: np.ndarray,
		model: DecisionTreeClassifier,
		y: np.ndarray = None,
		ypred: np.ndarray = None,
		feature_names: Iterable[str] = None,
		event_mapping: Dict[int, str] = None) -> NoReturn:

		self.X = X
		self.model = model
		self.y = y
		self.ypred = ypred
		self.feature_names = feature_names
		self.event_mapping = event_mapping

		self._parse_tree_structure()
		self._node_and_leaf_compute()

	def _parse_tree_structure(self) -> NoReturn:
		self.n_nodes = self.model.tree_.node_count
		self.children_left = self.model.tree_.children_left
		self.children_right = self.model.tree_.children_right
		self.feature = self.model.tree_.feature
		self.threshold = self.model.tree_.threshold
		self.n_node_samples = self.model.tree_.n_node_samples
		self.predicted_values = self.model.tree_.value

	def _node_and_leaf_compute(self) -> NoReturn:
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

	def node_and_leaf_print(self) -> NoReturn:
		print(
			'The binary tree structure has %s nodes and has '
			'the following tree structure:'
			.format(self.n_nodes)
		)
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

	def calculate_decision_paths(self, X: np.ndarray) -> NoReturn:
		self.node_indicator = self.model.decision_path(X)
		self.leave_id = self.model.apply(X)

	def all_decision_paths(
		self,
		use_readable_signs: bool = False,
		show_dominant_prediction: bool = True,
		show_n_samples: bool = True) -> Iterable[Iterable[str]]:
		''' Calculate all plain-English decision paths '''

		all_decision_paths = {}
		for idx in range(self.X.shape[0]):
			all_decision_paths[idx] = self.decision_path(
				idx,
				use_readable_signs,
				show_dominant_prediction,
				show_n_samples
			)
		return all_decision_paths

	def decision_path(
		self,
		sample_idx: int,
		use_readable_signs: bool = False,
		show_dominant_prediction: bool = True,
		show_n_samples: bool = True) -> Iterable[str]:
		''' Calculate plain-English decision path for a particular sample '''

		if not hasattr(self, 'node_indicator') or not hasattr(self, 'leave_id'):
			self.calculate_decision_paths(X)

		node_index = self.node_indicator.indices[
			self.node_indicator.indptr[sample_idx]:self.node_indicator.indptr[sample_idx + 1]
		]
		decisions = []
		
		if not ndarray_exists(self.ypred):
			prediction = self.model.predict(self.X[sample_idx].reshape(1, -1))[0]
		else:
			prediction = self.ypred[sample_idx]

		if self.event_mapping:
			prediction = self.event_mapping.get(*(prediction,)*2)

		for idx, node_idx in enumerate(node_index):
			if self.leave_id[sample_idx] == node_idx:
				continue

			# TODO: there's probably an incongruency between custom-threshold predictions
			# and the distribution of predictions using the built-in 50% threshold that would
			# be represented in self.predicted_values (an attribute that comes directly) from
			# the DecisionTreeClassifier
			if show_dominant_prediction:
				if isinstance(self.model, DecisionTreeClassifier):
					dominant_prediction = self.predicted_values[node_idx].argmax()
					if self.event_mapping:
						dominant_prediction = self.event_mapping.get(*(dominant_prediction,)*2)
				else:
					dominant_prediction = '{:.2f}'.format(self.predicted_values[node_idx].ravel()[0])
				dominant_prediction = ', dominant prediction: {}'.format(dominant_prediction)
			else:
				dominant_prediction = ''

			if show_n_samples:
				n_samples = str(self.n_node_samples[node_idx])
			else:
				n_samples = ''

			# If <PARTICULAR VALUE FOR SAMPLE> <= <DECISION THRESHOLD AT NODE>
			if self.X[sample_idx, self.feature[node_idx]] <= self.threshold[node_idx]:
				sign = '<='
			else:
				sign = '>'

			sign = READABLE_SIGN_MAPPING[sign] if use_readable_signs else sign

			decision = '`{}` ({:.2f}) {} {:.2f}'.format(
				self.feature_names[self.feature[node_idx]],
				self.X[sample_idx, self.feature[node_idx]],
				sign,
				self.threshold[node_idx]
			)

			decision = '{}{}{}{}{}'.format(
				# Indentation
				' ' * idx,
				# If not root node, use 'and' to indicate AND nature of decision tree
				' and ' if idx > 0 else '',
				# Actual, plain-English decision logic
				decision,
				# Number of samples at node
				' (samples: {})'.format(n_samples),
				# Dominant prediction at node
				dominant_prediction
			)

			decisions.append(decision)

		if self.event_mapping:
			final_prediction = '{} * Final prediction: {}'.format(' ' * idx, prediction)
			actual_outcome = self.event_mapping.get(self.y[sample_idx], self.y[sample_idx])
			final_prediction += '; actual outcome: {}'.format(actual_outcome) if ndarray_exists(self.y) else ''
		else:
			final_prediction = '{} * Final prediction: {:.2f}'.format(' ' * idx, prediction)
			actual_outcome = self.y[sample_idx]
			final_prediction += '; actual outcome: {:.2f}'.format(actual_outcome) if ndarray_exists(self.y) else ''

		decisions.append(final_prediction)
		final_decision_path = '\n'.join(decisions)
		return final_decision_path

	def plot_and_save(self) -> NoReturn:
		'''
		Use matplotlib to create graph and save to disk

		https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
		'''

		if self.model.max_depth is not None and self.model.max_depth > 6:
			warnings.warn(
				'This tree has a `max_depth` greater than 6 levels, '
				'meaning it could be fairly complex. The resolution '
				'for deep trees is subpar. An attempt will be made '
				'to save the tree to an image file, but for better '
				'resolution, try limiting the depth of your tree.'
			)
		
		export_graphviz(
			decision_tree=self.model,
			out_file='./model.dot',
			feature_names=self.feature_names,
			filled=True,
			rounded=True
		)

		# Convert from .dot to .png
		subprocess.call('dot -Tpng ./model.dot -o ./model.png && rm ./model.dot'.split())