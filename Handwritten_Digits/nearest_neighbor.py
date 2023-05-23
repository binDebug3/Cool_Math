import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy.stats import mode
from matplotlib import pyplot as plt



def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.
    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.
    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    
    # return the minimum of the norm of X - z
    return X[np.argmin(la.norm(X - z, axis=1))], np.min(la.norm(X - z, axis=1))



class KDTNode:
    """Node class for K-D Trees.
    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    
    def __init__(self, x):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        
        # raise a type error if the input is not a numpy array
        if type(x) is not np.ndarray:
            raise TypeError("Not an array")
        
        # set default attributes
        self.value = x
        self.pivot = None  # A reference to this node's pivot.
        self.left = None  # self.left.value < self.value
        self.right = None  # self.value < self.right.value



class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.
    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """

    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None


    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """

        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            
            else:
                return _step(current.right)         # Recursively search right.
        
            # Start the recursive search at the root of the tree.
        return _step(self.root)


    def insert(self, data):
        """Insert a new node containing the specified data.
        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.
        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        
        # isLeft = True
        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if np.allclose(data, current.value):
                raise ValueError(str(data) + " is already in the tree.")
            
            elif data[current.pivot] < current.value[current.pivot]:
                if current.left is None:
                    return current, True
                
                return _step(current.left)  # Recursively search left.
            
            else:
                if current.right is None:
                    # isLeft = False
                    return current, False
                
                return _step(current.right)  # Recursively search right.

        node = KDTNode(data)
        
        # if there are no nodes, insert at the root
        if self.root is None:
            node.pivot = 0
            self.root = node
            self.k = len(data)
        
        else:
            # otherwise, check dimension, then search for the parent with the step function
            if len(data) != len(self.root.value):
                raise ValueError("Data has incorrect dimension")
            
            parent, isLeft = _step(self.root)
            
            # add the node to the corresponding child position
            if isLeft:
                parent.left = node
            else:
                parent.right = node
            
            # set the pivot
            node.pivot = (parent.pivot + 1) % len(self.root.value)


    def query(self, z):
        """Find the value in the tree that is nearest to z.
        Parameters:
            z ((k,) ndarray): a k-dimensional target point.
        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """

        def KDSearch(current, nearest, dStar):
            # base case
            if current is None:
                return nearest, dStar
            
            # initialize search values
            x = current.value
            i = current.pivot
            
            # if the norm of x-z is less than d* update nearest and d*
            if la.norm(x - z) < dStar:
                nearest = current
                dStar = la.norm(x - z)
            
            # update nearest and d* recursively with the appropriate child
            if z[i] < x[i]:
                nearest, dStar = KDSearch(current.left, nearest, dStar)
                if z[i] + dStar >= x[i]:
                    nearest, dStar = KDSearch(current.right, nearest, dStar)
            
            else:
                nearest, dStar = KDSearch(current.right, nearest, dStar)
                if z[i] - dStar <= x[i]:
                    nearest, dStar = KDSearch(current.left, nearest, dStar)
           
            return nearest, dStar
        
        # return the results of the KD search starting from the root
        node, dStar = KDSearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, dStar


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.
        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        
        # convert the KDTree to its string representation as written by the instructors
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        
        # return the formatted string
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)



class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    
    def __init__(self, n_neighbors):
        # initialize with the number of neighbors, an empty tree, and empty labels
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None
    
    def fit(self, x, y):
        # create a new KDTree with x and set labels to y
        self.tree = KDTree(x)
        self.labels = y

    def predict(self, z):
        # query the KDTree based on z
        distances, indices = self.tree.query(z, k=self.n_neighbors)
        return mode(self.labels[indices])[0][0]



def digit_classifier(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.
    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.
    Returns:
        (float): the classification accuracy.
    """
    
    # get data and initialize vectors with the data
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float64)  # Training data
    y_train = data["y_train"]  # Training labels
    X_test = data["X_test"].astype(np.float64)  # Test data
    y_test = data["y_test"]  # Test labels
    
    # create a new classifier object and fit it with the training data
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    total = len(y_test)
    correct = 0
    
    # count the total number correct and return the percentage
    for i in range(0, total):
        if knn.predict(X_test[i]) == y_test[i]:
            correct += 1
    
    return correct / total

