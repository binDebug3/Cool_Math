import numpy as np
from scipy import linalg as la
import re



class MarkovChain:
    """A Markov chain with finitely many states.
    """

    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.
        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.
        Raises:
            ValueError: if A is not square or is not column stochastic.
        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """

        self.A = A
        dict = {}

        # check for error states
        if A.shape[0] != A.shape[1]:
            raise ValueError("Incorrect input: not square")

        for i in range(A.shape[1]):
            if round(sum(A[:, i])) != 1:
                raise ValueError("Incorrect input: not column stochastic")

        # update all the states
        if states is not None:
            for i, state in enumerate(states):
                dict.update({state: i})

        else:
            for i in range(0, A.shape[0]):
                dict.update({i: i})

        # set the attributes
        self.states = states
        self.dict = dict


    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.
        Parameters:
            state (str): the label for the current state.
        Returns:
            (str): the label of the state to transitioned to.
        """

        # return the transition state
        return self.states[np.argmax(np.random.multinomial(1, self.A[:, self.dict.get(state)]))]


    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.
        Parameters:
            start (str): The starting state label.
        Returns:
            (list(str)): A list of N state labels, including start.
        """

        # set parameters
        path = [start]
        current = start

        # walk the path from start to end of n steps
        for i in range(N - 1):
            current = self.transition(current)
            path.append(current)

        return path


    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.
        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.
        Returns:
            (list(str)): A list of state labels from start to stop.
        """

        # set variables
        path = [start]
        current = start

        # walk the path until the stop state
        while current is not stop:
            current = self.transition(current)
            path.append(current)

        return path


    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.
        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.
        Returns:
            ((n,) ndarray): The steady state distribution vector of A.
        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """

        # set variables
        count = 0
        x = np.random.random(self.A.shape[0])
        x /= la.norm(x)
        xOld = self.A @ x

        # recompute x until it converges or the count runs out
        while count < maxiter and la.norm(xOld - x, ord=1) > tol:
            xOld = x.copy()
            x = self.A @ x
            count += 1

        if count == maxiter:
            raise ValueError("The matrix does not converge")

        return x


class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.
    """

    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. 
        """

        # read the file
        with open(filename, "r") as text:
            yoda = text.read()

        unique = set(yoda.split())
        lines = yoda.split('\n')
        unique.add("$tart")
        unique.add("$top")

        # set the states attribute
        self.states = list(unique)

        # build the dictionary attribute
        yodaDict = {}
        for i, state in enumerate(self.states):
            yodaDict.update({state: i})
        self.dict = yodaDict

        # build the transition matrix attribute
        trans = np.zeros((len(unique), len(unique)))

        for line in lines:
            # get the words
            words = line.split()
            words = ["$tart",] + words + ["$top",]
            end = len(words)

            # increment the appropriate element
            for index in range(end):
                trans[yodaDict.get(words[index]), yodaDict.get(words[index - 1])] += 1
            trans[end - 1, end - 1] = 1

        # turn the transition matrix into a probability transition matrix
        trans /= trans.sum(axis=0, keepdims=1)
        self.A = trans



    def babble(self):
        """Create a random sentence using MarkovChain.path().
        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.
        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """

        # build a sentence using the path method
        crazy = self.path("$tart", "$top")
        return " ".join(crazy[1:-1])