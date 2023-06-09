from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread



def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A
    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.
    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    
    # get the eigenvectors and eigenvalues of AHA
    eigs, vecs = la.eig(A.conj().T @ A)
    
    # build sigma
    sigma = np.sqrt(eigs)
    indices = np.argsort(sigma)[::-1]
    sigma = sigma[indices]
    vecs = vecs[:,indices]
    sigma = np.array([x if abs(x) > tol else 0 for x in sigma])
    r = np.count_nonzero(sigma)
    sigma = sigma[:r]
    
    # build V
    V = vecs[:,:r]
    
    # build U
    U = (A @ V) / sigma
    
    # return the SVD Decomposition
    return U, sigma, V.conj().T



def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    
    # initialize matrices
    base = np.linspace(0, 2*np.pi, num=200)
    circle = [np.cos(base), np.sin(base)]
    e = [[1,0,0],[0,0,1]]
    
    # get the SVD of A
    U, W, V = la.svd(A)
    W = np.diag(W)
    
    # plot the original circle and lines
    plt.subplot(2,2,1)
    plt.plot(circle[0], circle[1])
    plt.plot(e[0], e[1])
    plt.xlabel("$S$")
    plt.axis("equal")
    
    # plot the rotated circle and lines
    plt.subplot(2,2,2)
    circleStep1 = V @ circle
    eStep1 = V @ e
    plt.plot(circleStep1[0], circleStep1[1])
    plt.plot(eStep1[0], eStep1[1])
    plt.xlabel("V^H S$")
    plt.axis("equal")
    
    # plot the rotated transposed circle and lines
    plt.subplot(2,2,3)
    circleStep2 = W @ circleStep1
    eStep2 = W @ eStep1
    plt.plot(circleStep2[0], circleStep2[1])
    plt.plot(eStep2[0], eStep2[1])
    plt.xlabel("$\Sigma V^H S$")
    plt.axis("equal")
   
    # plot the rotated transposed rotated circle and lines
    plt.subplot(2,2,4)
    circleStep3 = U @ circleStep2
    eStep3 = U @ eStep2
    plt.plot(circleStep3[0], circleStep3[1])
    plt.plot(eStep3[0], eStep3[1])
    plt.xlabel("$U\Sigma V^H S$")
    
    # show the graph
    plt.axis("equal")
    plt.tight_layout()
    plt.show()



def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.
    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.
    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    # get the SVD of A and compact it using s
    U, W, V = la.svd(A)
    
    if s > len(W):
        raise ValueError("Rank request is too big")
    
    UHat = U[:,:s]
    WHat = W[:s]
    VHat = V[:s,:]

    # calculate the number of elements to store and return compressed A
    size = UHat.size + WHat.size + VHat.size
    return UHat @ np.diag(WHat) @ VHat, size



def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.
    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.
    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    # get the SVD of A
    U, W, V = la.svd(A)
    
    # raise and error if the error is too low
    if err <= np.min(abs(W - err)):
        raise ValueError("You're too picky")
    
    # find the optimal s
    s = np.argmax(W < err)
    
    # compute the compact form of the SVD of A
    UHat = U[:, :s]
    WHat = W[:s]
    VHat = V[:s, :]
    
    # return the size and the compact SVD
    size = UHat.size + WHat.size + VHat.size
    return UHat @ np.diag(WHat) @ VHat, size



def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.
    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    
    # import the image and find the shape
    image = imread(filename) / 255.
    size = image.shape
    entries = image.size
    gray = False
    
    # if the picture is in color
    if len(size) == 3:
        # find the SVD approximation for each color based on s
        red, costRed = svd_approx(image[:,:,0], s)
        green, costGreen = svd_approx(image[:,:,1], s)
        blue, costBlue = svd_approx(image[:,:,2], s)
        
        red = np.clip(red, 0, 1)
        green = np.clip(green, 0, 1)
        blue = np.clip(blue, 0, 1)
        
        grayScale = np.dstack([red, green, blue])
        entries -= costRed + costBlue + costGreen
    
    else:
        # otherwise find the SVD approximate for the image based on s
        gray = True
        grayScale, cost = svd_approx(image, s)
        grayScale = np.clip(grayScale, 0, 1)
        entries -= cost
    
    if gray:
        # plot the original and the compressed grayscale images
        plt.subplot(1, 2, 1)
        plt.suptitle("Entries: " + str(image.size))
        plt.title("Original")
        plt.imshow(image, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title(f"Compressed ({s})")
        plt.suptitle("Entries: " + str(entries))
        plt.imshow(grayScale, cmap="gray")
    
    else:
        # plot the original and the compressed colored images
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.suptitle("Entries: " + str(image.size))
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title(f"Compressed ({s})")
        plt.suptitle("Entries: " + str(entries))
        plt.imshow(grayScale)
    
    plt.show()

