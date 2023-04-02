# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Name> Dallin Stewart
<Class> ACME 003
<Date> 10/21/22
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.
    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        # create a new dictionary
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        # return a string representation of the dicitionary
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.
        Parameters:
            n: the label for the new node.
        """
        # insert a node to the dictionary
        self.d.update({n: set()})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.
        Parameters:
            u: a node label.
            v: a node label.
        """
        # if u or v is not one of the nodes, add it to the graph
        if u not in self.d.keys():
            self.add_node(u)
        if v not in self.d.keys():
            self.add_node(v)
        # add an edge from u to v and v to u
        self.d.get(u).add(v)
        self.d.get(v).add(u)

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.
        Parameters:
            n: the label for the node to remove.
        Raises:
            KeyError: if n is not in the graph.
        """
        # if the node is not in the graph, raise an error
        if n not in self.d.keys():
            raise KeyError(n, "is not in the graph")
        # for each node in the graph, remove any edges to the node
        for key in self.d.keys():
            values = self.d.get(key)
            if n in values:
                values.remove(n)
        # remove the node from the graph
        self.d.pop(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.
        Parameters:
            u: a node label.
            v: a node label.
        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # if either node on the edge is not in the graph, raise an error
        if u not in self.d.keys() or v not in self.d.keys():
            raise KeyError("One or more nodes are not in the graph")
        elemRemove = False
        # check for the edge from u to v
        for value in self.d.get(u):
            if value == v:
                elemRemove = True
        # if no edge can be removed, raise an error
        if not elemRemove:
            raise KeyError("There is no edge between", u, "and", v)
        # remove the edge from u to v and v to u
        self.d.get(v).remove(u)
        self.d.get(u).remove(v)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.
        Parameters:
            source: the node to start the search at.
        Returns:
            (list): the nodes in order of visitation.
        Raises:
            KeyError: if the source node is not in the graph.
        """
        # if the source is not in the graph, raise an error
        if source not in self.d.keys():
            raise KeyError(source, "node is not in the graph")
        # make a deque for the nodes to parse and a list of visited nodes
        queued = deque(source)
        visited = []
        # while there are still nodes to visit
        while len(queued) > 0:
            # pop the first node in the deque and add it to visited
            currNode = queued.popleft()
            visited.append(currNode)
            # add all adjacent nodes that have not been visited to the deque
            adjNodes = self.d.get(currNode)
            for node in adjNodes:
                if node not in visited and node not in queued:
                    queued.append(node)
        # return the path
        return visited


    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.
        Parameters:
            source: the node to start the search at.
            target: the node to search for.
        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.
        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # if the source or the target is not in the graph
        if source not in self.d.keys() or target not in self.d.keys():
            raise KeyError(source, "or", target, "node is not in the graph")
        # if the source is the target, return the target
        if source == target:
            return [target]
        # make a deque for the nodes to parse and a list of visited nodes
        queued = deque([source])
        visited = []
        steps = {}
        # while there are still nodes to visit
        while len(queued) > 0:
            # pop the first node in the deque and add it to visited
            currNode = queued.popleft()
            visited.append(currNode)
            # add all adjacent nodes that have not been visited to the deque
            adjNodes = self.d.get(currNode)
            for node in adjNodes:
                if node not in visited and node not in queued:
                    steps.update({node: currNode})
                    queued.append(node)
                # if the node is the target, return the path to the node
                if node == target:
                    visited.append(node)
                    queued.clear()
                    break
        pathBack = [target]
        stepBack = target
        while stepBack is not source:
            stepBack = steps.get(stepBack)
            pathBack.append(stepBack)
        pathBack.reverse()
        return pathBack


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        # initialize attributes
        self.movies = set()
        self.actors = set()
        self.graph = nx.Graph()
        # get the data
        with open(filename, 'r', encoding="utf8") as movieData:
            info = movieData.readlines()
        # for each line, get the title and the list of actors and add them as attributes
        for movie in info:
            film = movie.strip().split("/")
            title = film[0]
            cast = film[1:]
            self.movies.add(title)
            self.graph.add_node(title)
            # add an edge from the movie to every cast member
            for member in cast:
                self.graph.add_edge(title, member)
                self.actors.add(member)

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # get the shortest path from every actor to the target
        path = nx.shortest_path(self.graph, source, target)
        length = 0
        # compute the path and its length
        for step in path:
            if step not in self.movies:
                length += 1
        return path, len(path) // 2

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.
        Returns:
            (float): the average path length from actor to target.
        """
        lengths = []
        # get all shortest paths
        all_short_paths = nx.shortest_path_length(self.graph, target)
        # build a list of lengths
        for actor in self.actors:
            lengths.append(all_short_paths.get(actor) // 2)
        # plot length data
        plt.hist(lengths, bins=[i-.5 for i in range(8)])
        plt.xlabel("Distance to Kevin Bacon")
        plt.ylabel("Number of Actors")
        plt.title("Everyone Loves Kevin Bacon")
        plt.show()
        # return average length
        return sum(lengths) / len(lengths)



if __name__ == "__main__":
    pass
    # graph = Graph()
    # graph.add_node(1)
    # graph.add_node(2)
    # graph.add_node(5)
    #
    # print(graph)
    #
    # graph.add_edge(1, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(1, 3)
    #
    # print(graph)
    #
    # graph.remove_edge(2, 3)
    # print(graph)

    # graph = Graph()
    # graph.add_node('A')
    # graph.add_node('B')
    # graph.add_node('C')
    # graph.add_node('D')
    #
    # print(graph)
    #
    # graph.add_edge('A','B')
    # graph.add_edge('A','D')
    # graph.add_edge('B','D')
    # graph.add_edge('D','C')
    #
    # # print(graph.traverse('Q'))
    #
    # print(graph)

    # graph = Graph()
    # graph.add_node(1)
    # graph.add_node(2)
    # graph.add_node(3)
    # graph.add_node(4)
    # graph.add_node(5)
    # graph.add_node(6)
    #
    # print(graph)
    #
    # graph.add_edge(1,2)
    # graph.add_edge(2,3)
    # graph.add_edge(2,4)
    # graph.add_edge(2,5)
    # graph.add_edge(3,4)
    # graph.add_edge(5,4)
    # graph.add_edge(5,6)
    # graph.add_edge(7,3)
    # graph.add_edge(7,4)
    #
    # print(graph)
    #
    # graph.remove_edge(2,4)
    #
    # print(graph.shortest_path(1,4))


    # test problem 1
    # graph = Graph()
    # graph.add_node("a")
    # graph.add_node("b")
    # graph.add_node("c")
    # graph.add_edge("a", "b")
    # graph.add_edge("a", "c")
    # graph.remove_edge("a", "c")
    # # graph.remove_edge("a", "c")
    # graph.remove_node("c")
    # graph.remove_node("c")
    # print(graph)

    # test problem 2
    # graph = Graph()
    # graph.add_node("A")
    # graph.add_node("B")
    # graph.add_node("C")
    # graph.add_node("D")
    # graph.add_edge("A", "D")
    # graph.add_edge("A", "B")
    # graph.add_edge("B", "A")
    # graph.add_edge("B", "D")
    # graph.add_edge("C", "D")
    # graph.add_edge("D", "B")
    # graph.add_edge("D", "A")
    # graph.add_edge("D", "C")
    # print(graph)
    # print(graph.traverse("A"))

    # test problem 3
    # graph = Graph()
    # graph.add_node("a")
    # graph.add_node("b")
    # graph.add_node("c")
    # graph.add_node("d")
    # graph.add_node("e")
    # graph.add_edge("a", "b")
    # graph.add_edge("a", "c")
    # graph.add_edge("c", "e")
    # graph.add_edge("c", "d")
    # print(graph.shortest_path("A", "C"))

    # test problem 4
    data = MovieGraph("movie_data.txt")

    # test problem 5
    # print(data.path_to_actor("Kevin Bacon", "Tom Cruise"))
    print(data.path_to_actor("Mark Hamill", "Kevin Bacon"))

    # test problem 6
    # print(data.average_number("Kevin Bacon"))
