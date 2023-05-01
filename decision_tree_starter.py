# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO: implement information gain function
        entropy_before = DecisionTree.entropy(y)
        left_X_idx = np.where(X <= thresh)
        left_y = y[left_X_idx]
        right_X_idx = np.where(X > thresh)
        right_y = y[right_X_idx]
        
        left_entropy = DecisionTree.entropy(left_y)
        right_entropy = DecisionTree.entropy(right_y)
        
        
        n = len(y) 
        left_wt = len(left_y)/n
        right_wt = len(right_y)/n
        
        entropy_after = left_wt * left_entropy + right_wt * right_entropy
        info_gain = entropy_before - entropy_after
        
        return info_gain
    
    @staticmethod
    def entropy(y):
        labels = np.unique(y)
        entropy = 0
        for i in labels:
            pi = np.sum(y == i) / len(y)
            entropy += - pi * np.log2(pi)
        return entropy


    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [DecisionTree(max_depth=5, feature_labels = self.feature_labels) for i in range(self.n)]
        

    def fit(self, X, y):
        # TODO: implement function
        for tree in self.decision_trees:
            index = np.arange(len(y))
            size = int(.63 * len(y))
            sample_index = np.random.choice(index, size, replace = True)
            f_labels = y[sample_index]
            train_data = X[sample_index]
            if self.m != None:
                sub_feature_idx = np.random.choice(range(train_data.shape[1]), self.m, replace = False)
                train_data = train_data[:, sub_feature_idx]
            tree.fit(train_data, f_labels)
            

    def predict(self, X):
        # TODO: implement function
        prediction = np.array([])
        for tree in self.decision_trees:
            prediction = np.append(prediction, tree.predict(X))
        prediction = np.reshape(prediction,(len(self.decision_trees),len(X)))
        prediction = stats.mode(prediction, keepdims = False)[0]
        return prediction


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m = None, depth = None):
        if params is None:
            params = {}
        # TODO: implement function
        self.params = params
        self.n = n
        self.m = m
        self.max_depth = depth
        self.decision_trees = [DecisionTree(max_depth = depth) for i in range(self.n)]


# class BoostedRandomForest(RandomForest):
#     def fit(self, X, y):
#         self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
#         self.a = np.zeros(self.n)  # Weights on decision trees
#         # TODO: implement function
#         return self

#     def predict(self, X):
#         # TODO: implement function
#         pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)
        
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission.csv', index_label='Id')

    
def train_valid_split(data, label, num_valid):
    np.random.seed(1338)
    shuffled_indices = np.random.permutation(len(data))
    valid_indices = shuffled_indices[:num_valid]
    train_indices = shuffled_indices[num_valid:]
    
    return (data[valid_indices], label[valid_indices], data[train_indices],label[train_indices])

def bfs(tree):
    
    queue = [tree]
    node_num = 1
    
    while len(queue) > 0:
        cur_node = queue.pop(0)
        
        if cur_node.left is not None:
            queue.append(cur_node.left)
            
        if cur_node.right is not None:
            queue.append(cur_node.right)
    
        if cur_node.left is not None or cur_node.right is not None:
            print('Node: ', node_num)
            print(cur_node.features[cur_node.split_idx])
            print(cur_node.thresh)
            print('')
        else:
            print('Node: ' ,node_num)
            print("Label: ", cur_node.pred)
            print('')
            
        node_num += 1


def load(dataset):
    if dataset == "titanic":
        # Load titanic data
        path_train = '../dataset/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None, encoding=None)
        path_test = '../dataset/titanic/titanic_test_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None, encoding=None)
        y = data[1:, -1]  # label = survived
        class_names = ["Died", "Survived"]
        labeled_idx = np.where(y != '')[0]

        y = np.array(y[labeled_idx])
        y = y.astype(float).astype(int)


        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, :-1], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, :-1]) + onehot_features
        all_data = X
        all_label = y
        num_valid = int(.2 * len(all_label))
        valid_data, valid_label, train_data, train_label = train_valid_split(all_data, all_label, num_valid)

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = '../dataset/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        all_data = data['training_data']
        all_label = np.squeeze(data['training_labels'])
        num_valid = int(.2 * len(all_label))
        valid_data, valid_label, train_data, train_label = train_valid_split(all_data, all_label, num_valid)

        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
        
    print("Train/valid/test size:", train_data.shape, valid_data.shape, Z.shape)
    print(features)
        
    return all_data , all_label, valid_data, valid_label, train_data, train_label, features, Z 

#     print("Features:", features)
#     print("Train/valid/test size:", train_data.shape, valid_data.shape, Z.shape)

def model(tree, depth, all_data , all_label, valid_data, valid_label, train_data, train_label, features, purpose, test_data, num_subfeatures = None ):
    if tree == 'dTree':
    # Basic decision tree
        print("Model: Decision Tree")
        dt = DecisionTree(max_depth=depth, feature_labels=features)
    elif tree == 'rf':
        print("Model: Random Forest")
        dt = RandomForest(depth = depth, m = num_subfeatures)
        
    if purpose == 'train':
        dt.fit(train_data, train_label)

        prediction_train = dt.predict(train_data)
        #     print("Predictions Train ", prediction_train)
        accuracy_train = np.sum(train_label == prediction_train)/ len(prediction_train)
        print("Accuracy_Train",accuracy_train)

        prediction_valid = dt.predict(valid_data)
        #     print("Predictions Valid ", prediction_valid)
        accuracy_valid = np.sum(valid_label == prediction_valid)/ len(prediction_valid)
        print("Accuracy_Valid ", accuracy_valid)
    
    elif purpose == 'test':
        
        dt.fit(all_data, all_label)
        prediction_test = dt.predict(test_data)
    #     print("Predictions Test ", prediction_test[:100])
        results_to_csv(prediction_test)
        accuracy_valid = 0
    return accuracy_valid