import numpy as np
import networkx as nx
import itertools



class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.
    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """

    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.
        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """

        # check parameters
        n, m = A.shape
        if labels is None:
            labels = np.arange(n)

        if n != m or n != len(labels):
            raise ValueError("The number of labels is not equal to the number of nodes in the graph")

        # remove sinks
        for i in range(m):
            if np.sum(A[:,i]) == 0:
                A[:, i] = 1

        # normalize A and update attributes
        A = A / np.sum(A, axis=0)
        self.A = A
        self.labels = labels
        self.n = len(labels)


    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.
        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # compute the PageRank vector using the linear system method.
        altA = np.eye(self.n) - epsilon * self.A
        b = (1 - epsilon) / self.n * np.ones(self.n)
        p = np.linalg.solve(altA, b)

        # return dictionary mapping labels to vectors
        return {self.labels[i]: p[i] for i in range(self.n)}


    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.
        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # compute B
        B = epsilon * self.A + (1 - epsilon) / self.n * np.ones((self.n, self.n))

        # solve for eigenvectors
        val, vec = np.linalg.eig(B)
        p = vec[:, 0] / np.sum(vec[:, 0])

        # return dictionary mapping labels to vectors
        return {self.labels[i]: p[i] for i in range(self.n)}

    
    def _pNext(self, currP, ep):
        return ep * self.A @ currP + (1 - ep) / self.n * np.ones(self.n)


    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.
        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.
        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # compute initial values
        p0 = np.ones(self.n) / self.n
        p1 = self._pNext(p0, epsilon)
        t = 0

        # iteratively update p
        while t < maxiter:
            p1 = self._pNext(p0, epsilon)
            if np.linalg.norm(p1 - p0, ord=1) < tol:
                break
            p0 = p1
            t += 1

        # return the dictionary, even if it did not converge enough
        return {self.labels[i]: p1[i] for i in range(self.n)}



def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.
    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.
    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """

    # create a list of types of keys and values
    pairs = [(label, pagerank) for label, pagerank in d.items()]

    # sort the list
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # return the sorted list of labels
    return [label for label, pagerank in sorted_pairs]



def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.
    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.
    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.
    Returns:
        (list(str)): The ranked list of webpage IDs.
    """

    # read in data
    with open(filename, 'r') as name:
        data = name.read()
    data = data.split("\n")[:-1]

    # get a list of all possible sites, including sinks
    sites = set()
    for line in data:
        for ids in line.split("/"):
            sites.add(ids)

    sites = list(sites)
    sites.sort()
    n = len(sites)

    # create a dictionary that maps a site to its index
    site_dict = {sites[i]: i for i in range(n)}

    # create dictionary that maps a site to all the sites it links to
    link_dict = {line.split("/")[0]: line.split("/")[1:] for line in data}
    # including sinks
    for l in sites:
        if l not in link_dict.keys():
            link_dict.update({l: ""})

    # create transition matrix
    transition = np.zeros((n, n))
    for desde in sites:
        for hasta in link_dict[desde]:
            transition[site_dict[hasta], site_dict[desde]] += 1

    # rank pages and return sorted list
    pageRank = DiGraph(transition, sites)
    indices = get_ranks(pageRank.itersolve(epsilon=epsilon))
    return indices



def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks(
    Each line of the file has the format
        A,B
    meaning team A defeated team B.
    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.
    Returns:
        (list(str)): The ranked list of team names.
    """

    # read in data
    with open(filename, 'r') as name:
        data = name.read()
    data = data.split("\n")[1:-1]

    # get a list of all possible teams, including sinks
    teams = set()
    for line in data:
        for ids in line.split(","):
            teams.add(ids)
    teams = list(teams)
    teams.sort()
    n = len(teams)

    # create a dictionary that maps a team to its index
    teamMap = {teams[i]: i for i in range(n)}

    # build transition matrix
    transition = np.zeros((n, n))
    for game in data:
        comps = game.split(",")
        transition[teamMap[comps[0]], teamMap[comps[1]]] += 1

    # rank teams and return sorted list
    teamRank = DiGraph(transition, teams)
    indices = get_ranks(teamRank.itersolve(epsilon=epsilon))
    return [i for i in indices]



def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks()
    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """

    # read in data
    with open(filename, 'r', encoding="utf-8") as file:
        data = file.read()
    full_movies = data.split("\n")
    cast = {movie.split("/")[0]: movie.split("/")[1:] for movie in full_movies}

    # get a list of all possible actors, including sinks
    actors = set()
    for movie in full_movies:
        for ids in movie.split("/")[1:]:
            actors.add(ids)
    actors = list(actors)
    actors.sort()
    n = len(actors)

    # create a dictionary that maps an actor to their index
    actorMap = {actors[i]: i for i in range(n)}

    # construct network
    DG = nx.DiGraph()
    for title in cast.keys():
        for pair in itertools.combinations(cast[title], 2):
            if DG.has_edge(pair[1], pair[0]):
                DG[pair[1]][pair[0]]["weight"] += 1
            else:
                DG.add_edge(pair[1], pair[0], weight=1)

    # return sorted list
    return get_ranks(nx.pagerank(DG, alpha=epsilon))