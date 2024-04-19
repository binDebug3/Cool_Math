import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # get all the unique words in X
        all_unique = X.str.split().explode().unique()
        self.spam_probs = {word: 0 for word in all_unique}
        self.ham_probs = {word: 0 for word in all_unique}

        # initialize the length variables
        nspam = y.value_counts()['spam']
        nham = y.value_counts()['ham']
        nsamples = len(y)

        # compute P(C=Ham) and P(C=Spam)
        self.prob_spam = nspam / nsamples
        self.prob_ham = nham / nsamples

        # get the spam and ham words
        spam = X[y=='spam'].str.split().explode()
        ham = X[y=='ham'].str.split().explode()

        # get the length (+2 for smoothing) and the value counts for each set
        self.spam_length = len(spam) + 2
        self.ham_length = len(ham) + 2
        spam = spam.value_counts()
        ham = ham.value_counts()

        # optimize
        init_spam = 1 / self.spam_length
        init_ham = 1 / self.ham_length

        def compute_prob(value, spam=True):
            if spam:
                return (value + 1) / self.spam_length
            return (value + 1) / self.ham_length
        
        # compute P(x_i|C)
        for word in all_unique:
            self.spam_probs[word] = compute_prob(spam.loc[word]) if word in spam.index else init_spam
            self.ham_probs[word] = compute_prob(ham.loc[word], spam=False) if word in ham.index else init_ham

        return self



    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # initialize the return array
        ret = np.zeros((len(X), 2))

        for i, row in enumerate(X):
            # initialize the sums
            ham_sum = 0
            spam_sum = 0

            for word in row.split():
                # compute the sum of the log probabilities
                try:
                    ham_sum += np.log(self.ham_probs[word]) if word in self.ham_probs else np.log(1/2)
                    spam_sum += np.log(self.spam_probs[word]) if word in self.spam_probs else np.log(1/2)
                # catch the warning for log(0)
                except RuntimeWarning:
                    print(f'{word} is missing from the dictionary')
            
            # add the log probabilities to the return array
            ret[i, 0] = np.log(self.prob_ham) + ham_sum
            ret[i, 1] = np.log(self.prob_spam) + spam_sum
        
        return ret



    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # return the labels based on the computed probabilities
        return np.array(['spam' if row[1] > row[0] else 'ham' for row in self.predict_proba(X)])



def train_nb():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data into the messages and labels
    X = df.Message
    y = df.Label

    # create the train-test split
    x_test, x_train, y_test, y_train = train_test_split(X, y, test_size=0.2)

    # initialize and fit the model
    nb = NaiveBayesFilter()
    nb.fit(x_train, y_train)

    # compare predictions to y_test
    spam_correct = 0
    ham_incorrect = 0
    for i, pred in enumerate(nb.predict(x_test)):
        if pred == 'spam' and y_test.iloc[i] == 'spam':
            spam_correct += 1
        elif pred == 'ham' and y_test.iloc[i] == 'spam':
            ham_incorrect += 1
    
    # compute the proportions and return
    spam_correct /= len(y_test[y_test=='spam'])
    ham_incorrect /= len(y_test[y_test=='ham'])

    return spam_correct, ham_incorrect
    
    



class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # get all the unique words in X
        all_unique = X.str.split().explode().unique()
        self.spam_rates = {word: 0 for word in all_unique}
        self.ham_rates = {word: 0 for word in all_unique}

        # initialize the length variables
        nspam = y.value_counts()['spam']
        nham = y.value_counts()['ham']
        nsamples = len(y)

        # compute P(C=Ham) and P(C=Spam)
        self.prob_spam = nspam / nsamples
        self.prob_ham = nham / nsamples

        # get the spam and ham words
        spam = X[y=='spam'].str.split().explode()
        ham = X[y=='ham'].str.split().explode()

        # get the length (+2 for smoothing) and the value counts for each set
        self.spam_length = len(spam) + 2
        self.ham_length = len(ham) + 2
        spam = spam.value_counts()
        ham = ham.value_counts()

        # optimize
        init_spam = 1 / self.spam_length
        init_ham = 1 / self.ham_length

        def compute_prob(value, spam=True):
            if spam:
                return (value + 1) / self.spam_length
            return (value + 1) / self.ham_length
        
        # compute P(x_i|C)
        for word in all_unique:
            self.spam_rates[word] = compute_prob(spam.loc[word]) if word in spam.index else init_spam
            self.ham_rates[word] = compute_prob(ham.loc[word], spam=False) if word in ham.index else init_ham

        # compute r_{i,k}??

        return self



    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        ret = np.zeros((len(X), 2))

        for i, row in enumerate(X):
            # initialize the word data
            row = row.split()
            words, count = np.unique(row, return_counts=True)

            # initialize the sums
            ham_sum = np.log(self.prob_ham)
            spam_sum = np.log(self.prob_spam)

            # optimize
            length = len(row)
            init_spam = 1 / self.spam_length
            init_ham = 1 / self.ham_length

            def poisson(message_count, value):
                # returns the log of the poisson pmf
                return stats.poisson.logpmf(message_count, value * length)
            
            for word in words:
                # find word count in the message
                mc = count[np.where(words==word)[0][0]]
                try:
                    # compute the sums
                    ham_sum += poisson(mc, self.ham_rates[word]) if word in self.ham_rates else poisson(mc, init_ham)
                    spam_sum += poisson(mc, self.spam_rates[word]) if word in self.spam_rates else poisson(mc, init_spam)
                except RuntimeWarning:
                    # catch the warning for log(0)
                    print(f'{word} is missing from the dictionary')
            
            # add the log probabilities to the return array
            ret[i, 0] = ham_sum
            ret[i, 1] = spam_sum
        return ret



    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # return the labels based on the computed probabilities
        return np.array(['spam' if row[1] > row[0] else 'ham' for row in self.predict_proba(X)])


def train_bf():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data into the messages and labels
    X = df.Message
    y = df.Label

    # create the train-test split
    x_test, x_train, y_test, y_train = train_test_split(X, y, test_size=0.2)

    # initialize and fit the model
    pb = NaiveBayesFilter()
    pb.fit(x_train, y_train)

    # compare predictions to y_test
    spam_correct = 0
    ham_incorrect = 0
    for i, pred in enumerate(pb.predict(x_test)):
        if pred == 'spam' and y_test.iloc[i] == 'spam':
            spam_correct += 1
        elif pred == 'ham' and y_test.iloc[i] == 'spam':
            ham_incorrect += 1
    
    # compute the proportions and return
    spam_correct /= len(y_test[y_test=='spam'])
    ham_incorrect /= len(y_test[y_test=='ham'])

    return spam_correct, ham_incorrect
    


def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # initialize model parameters
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)
    test_counts = vectorizer.transform(X_test)

    # initialize and fit the model
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)

    # test model and return results
    test_counts = vectorizer.transform(X_test)
    return clf.predict(test_counts)



if __name__ == "__main__":
    pass
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data into the messages and labels
    X = df.Message
    y = df.Label

    # NAIVE BAYES
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])

    print(nb.ham_probs['out'])
    print(nb.spam_probs['out'])

    print("ln(P(C=k,x)) for each x in X and for each class")
    print(nb.predict_proba(X[800:805]))

    print("Predict the labels of each row in X, using self.predict_proba()")
    print(nb.predict(X[800:805]))

    print("The proportion of spam that were correctly identified and the proportion of ham that were incorrectly identified")
    print(train_nb())


    # POISSON
    pb = PoissonBayesFilter()
    pb.fit(X[:300], y[:300])
    
    print(pb.ham_rates['in'])
    print(pb.spam_rates['in'])

    print("ln(P(C=k,x)) for each x in X and for each class")
    print(pb.predict_proba(X[800:805]))

    print("Predict the labels of each row in X, using self.predict_proba()")
    print(pb.predict(X[800:805]))

    print("The proportion of spam that were correctly identified and the proportion of ham that were incorrectly identified")
    print(train_bf())

    print(sklearn_naive_bayes(X[:300], y[:300], X[800:805]))


