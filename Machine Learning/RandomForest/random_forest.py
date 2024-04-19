import time
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier



class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""


    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]


    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        
        # return the answer to the question
        return sample[self.column] >= self.value


    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))



def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # find the left and right partitions
    left = data[data[:,question.column] >= question.value]
    right = data[data[:,question.column] < question.value]

    # reshape the arrays if they are empty
    if left.size == 0:
        left.reshape(0, data.shape[1])
    if right.size == 0:
        right.reshape(0, data.shape[1])
    
    # return the partitions
    return left, right



# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]



# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts



# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)



def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    # initialize local variables
    best_gain = 0
    best_question = None

    # choose a random subset of features to consider
    if random_subset:
        num_features = int(np.floor(np.sqrt(data.shape[1]-1)))
        random_features = np.random.choice(data.shape[1]-1, num_features, replace=False)
    else:
        random_features = range(data.shape[1]-1)

    for col in random_features:
        for row in range(data.shape[0]):

            # initialize the question and split the data
            question_iter = Question(col, data[row,col], feature_names)
            left_iter, right_iter = partition(data, question_iter)

            # skip if the split is invalid
            if num_rows(left_iter) < min_samples_leaf or num_rows(right_iter) < min_samples_leaf:
                continue

            # calculate the gain and update the best gain and question
            gain = info_gain(data, left_iter, right_iter)
            if gain >= best_gain:
                best_gain = gain
                best_question = question_iter

    # return the best gain and question
    return best_gain, best_question



class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        # initialize attributes
        self.question = question
        self.left = left_branch
        self.right = right_branch



def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    # base case
    if data.shape[0] < 2*min_samples_leaf or current_depth >= max_depth:
        return Leaf(data)

    # recursive case
    optimal_gain, optimal_question = find_best_split(data, feature_names, min_samples_leaf, random_subset)
    if optimal_gain == 0:
        return Leaf(data)

    # partition the data and return the branches recursively
    left_p, right_p = partition(data, optimal_question)
    left_branch = build_tree(left_p, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)
    right_branch = build_tree(right_p, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)

    return Decision_Node(optimal_question, left_branch, right_branch)



def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # base case
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)

    # recursive case for left and right branches
    if my_tree.question.match(sample):
        return predict_tree(sample, my_tree.left)
    else:
        return predict_tree(sample, my_tree.right)



def analyze_tree(dataset, my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    # count the number of correct predictions
    correct = sum(predict_tree(dataset[row,:], my_tree) == dataset[row,-1] for row in range(dataset.shape[0]))

    # return the proportion of correct predictions
    return correct / dataset.shape[0]



def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    # predict the label for each tree in the forest
    labels = [predict_tree(sample, tree) for tree in forest]

    # return the most common label
    return max(labels, key=labels.count)



def analyze_forest(dataset, forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    # count the number of correct predictions
    correct = sum(predict_forest(dataset[row, :], forest) == dataset[row, -1] for row in range(dataset.shape[0]))

    # return the proportion of correct predictions
    return correct / dataset.shape[0]



def build_parkinsons_tree():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    # load data
    park_data = np.loadtxt('parkinsons.csv', delimiter=',')[:,1:]
    park_features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str, comments=None)

    # randomly select 130 samples
    np.random.seed(0)
    random_indices = np.random.choice(park_data.shape[0], 130, replace=False)
    data = park_data[random_indices, :]
    train_park = data[:100, :]
    test_park = data[100:, :]

    # split whole dataset into 80% train and test
    shuffled = np.random.permutation(data)
    split = int(0.8 * data.shape[0])
    train_sk = shuffled[:split, :]
    test_sk = shuffled[split:, :]

    # run my forest
    start_mine = time.perf_counter()
    park_forest = [build_tree(train_park, park_features, min_samples_leaf=15, max_depth=4, random_subset=True) for _ in range(10)]
    accuracy_mine = analyze_forest(test_park, park_forest)
    delay_mine = time.perf_counter() - start_mine

    # run sklearn's forest on small set
    start_sk = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    forest.fit(train_park[:,:-1], train_park[:,-1])
    accuracy_sk = forest.score(test_park[:,:-1], test_park[:,-1])
    delay_sk = time.perf_counter() - start_sk

    # run sklearn's forest on whole set
    start_all = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    forest.fit(train_sk[:, :-1], train_sk[:, -1])
    accuracy_all = forest.score(test_sk[:, :-1], test_sk[:, -1])
    delay_all = time.perf_counter() - start_all

    # return the three tuples
    return (accuracy_mine, delay_mine), (accuracy_sk, delay_sk), (accuracy_all, delay_all)



# Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    # If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: # If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id



def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree, leaf_class=leaf_class)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)


if __name__ == "__main__":
    pass
    parent_path = ""
    animals = np.loadtxt(parent_path + 'animals.csv', delimiter=',')
    features = np.loadtxt(parent_path + 'animal_features.csv', delimiter=',', dtype=str, comments=None)
    names = np.loadtxt(parent_path + 'animal_names.csv', delimiter=',', dtype=str)
    

    # test Question class
    test_question = Question(column=1, value=3, feature_names=features)
    left, right = partition(animals, test_question)
    print(len(left), len(right))
    
    test_question = Question(column=1, value=75, feature_names=features)
    left, right = partition(animals, test_question)
    print(len(left), len(right))
    

    # test find best split
    print("Best gain and question:", find_best_split(animals, features))


    # build a tree!
    my_tree_3 = build_tree(animals, features)
    draw_tree(my_tree_3)
    print("See the pdf for the tree visualization")
    

    # shuffle and split the data
    animals_shuffled = animals.copy()
    np.random.shuffle(animals_shuffled)
    train = animals_shuffled[:80]
    test = animals_shuffled[80:]
    
    # train and analyze the tree
    my_tree_4 = build_tree(train, features)
    print("Accuracy:", analyze_tree(test, my_tree_4))


    # train and analyze the forest
    my_forest = [build_tree(train, features, random_subset=True) for _ in range(10)]
    print("Forest Accuracy:", analyze_forest(test, my_forest))

    # build the forest
    print(build_parkinsons_tree())

