import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sla
from imageio import imread
from matplotlib import pyplot as plt
import math



def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.
    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    
    # initialize variables
    n = A.shape[0]
    d = np.zeros((n, n), dtype=float)
    diags = np.sum(A, axis=1)
    
    # build diagonals
    for i in range(0, n):
        d[i, i] = diags[i]
    
    # return laplacian
    return d - A



def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.
    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.
    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    
    # get eigenvalues
    eigs = np.real(la.eigvals(laplacian(A)))
    
    # count connectedness and find algebraic multiplicity
    for i in range(0, eigs.size):
        # check for tolerance and minimum eigenvalue
        if abs(eigs[i]) < tol:
            eigs[i] = 0
    
    # return count and multiplicity
    return abs(np.count_nonzero(eigs) - len(eigs)), np.sort(eigs)[1]



# Helper function
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.
    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.
    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """

    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col) ** 2 + (Y - row) ** 2))
    mask = R < radius

    return (X[mask] + Y[mask] * width).astype(np.int), R[mask]



class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        """Write the constructor so that it accepts the name of an image file. 
        1. Read the image, 
        2. scale it so that it contains floats between 0 and 1, 
        3. then store it as an attribute. 
        4. If the image is in color, compute its brightness matrix by averaging the RGB values at each pixel (if it is
            a grayscale image, the image array itself is the brightness matrix). 
        5. Flatten the brightness matrix into a 1-D array and 
        6. store it as an attribute"""

        # read image and convert to 0 to 1
        self.image = imread(filename) / 255.
        self.size = self.image.shape
        self.gray = False

        # get the grayscale if appropriate
        if len(self.size) == 3:
            self.grayScale = np.ravel(self.image.mean(axis=2))

        else:
            self.gray = True
            self.grayScale = np.ravel(self.image)

    def show_original(self):
        """Display the original image."""

        # show full color or grayscale image as appropriate
        if self.gray:
            plt.imshow(self.image, cmap="gray")

        else:
            plt.imshow(self.image)

        plt.axis("off")
        plt.show()


    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

        # get the size and initialize matrices
        size = self.grayScale.shape[0]
        A = sparse.lil_matrix((size, size))
        D = np.empty(size)

        for i in range(0, size):
            # get the neighbors and the distances to index i
            vertices, distances = get_neighbors(i, r, self.size[0], self.size[1])

            # compute weights to update A and D
            f = self.grayScale[i]
            g = self.grayScale[vertices]
            values = np.exp(-abs(f - g) / sigma_B2 - distances / sigma_X2)

            A[i, vertices] = values
            D[i] = np.sum(values)

        return sparse.csc_matrix(A), D


    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""

        L = sparse.csgraph.laplacian(A)

        # compute the square root of D
        Dhalf = sparse.diags(1 / np.sqrt(D)).tocsc()

        # compute the eigenvalues and eigenvectors for the mask
        eigs, vec = sla.eigsh(Dhalf @ L @ Dhalf, which="SM", k=2)

        # return the vectors reshaped
        return vec[:,1].reshape(self.size[:2]) > 0


    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""

        # get adjacency matrix and find the mask
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)

        # plot the two cuts and the original image
        if self.gray:
            plt.subplot(1,3,1)
            plt.imshow(self.image*mask, cmap="gray")
            plt.subplot(1,3,2)
            plt.imshow(self.image*~mask, cmap="gray")

        else:
            # plot full color
            bigmask = np.dstack((mask, mask, mask))
            plt.subplot(1, 3, 1)
            plt.imshow(self.image*bigmask)
            plt.subplot(1, 3, 2)
            plt.imshow(self.image*~bigmask)

        plt.subplot(1,3,3)
        self.show_original()