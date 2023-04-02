import numpy as np
import sympy as sy
from scipy import linalg as la
from numpy import linalg as npla
from matplotlib import pyplot as plt



def plot_perturbations():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    # initialize local variables
    w_roots = np.arange(1, 21)
    n = len(w_coeffs)
    abs_cond = []
    rel_cond = []

    for i in range(100):
        # construct perturbations
        r = np.array([np.random.normal(1, 10e-10) for j in range(n)])

        # compute the new roots
        new_coeffs = np.multiply(w_coeffs, r)
        new_roots = np.roots(np.poly1d(new_coeffs))
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)

        # plot the roots
        x = np.real(new_roots)
        y = np.imag(new_roots)
        plt.scatter(x, y, 1, marker=',', alpha=0.9, c="k")

        # compute condition
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(r, np.inf)
        abs_cond.append(k)
        rel_cond.append(k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))

    # set plot parameters
    plt.scatter(x, y, 1, marker=',', alpha=0.9, c="k", label="Perturbed")
    plt.scatter(w_roots, np.zeros(len(w_roots)), c="b", label="Original")
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.title("Problem 2")
    plt.legend()
    plt.show()

    # compute and return mean
    mean_abs = np.mean(np.array(abs_cond))
    mean_rel = np.mean(np.array(rel_cond))
    return mean_abs, mean_rel



# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    # I didn't write this code so don't doc me for comments pls
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]



def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.
    Parameters:
        A ((n,n) ndarray): A square matrix.
    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """

    # construct perturbations
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j * imags

    # find eigenvalues
    L = la.eigvals(A)
    Lt = reorder_eigvals(L, la.eigvals(A + H))

    # compute condition numbers and return them
    kHat = npla.norm(L - Lt, ord=2) / npla.norm(H, ord=2)
    k = npla.norm(A, ord=2) * kHat / npla.norm(L, ord=2)
    return kHat, k



def condition_number(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # set up meshgrid
    xGrid = np.linspace(domain[0], domain[1], res)
    yGrid = np.linspace(domain[2], domain[3], res)
    mesh = np.zeros((len(xGrid), len(yGrid)))

    # compute condition number at each point
    for i, x in enumerate(xGrid):
        for j, y in enumerate(yGrid):
            mesh[i, j] = eig_cond(np.array([[1, x], [y, 1]]))[1]

    # set plot parameters
    plt.pcolormesh(mesh, cmap="gray_r")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title("Problem 4")
    plt.colorbar()
    plt.show()



def least_squares(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.
    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.
    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """

    # load data
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n + 1)

    # compute least squares approximation
    ls1 = la.inv(A.T @ A) @ A.T @ yk
    q, r = la.qr(A, mode='economic')
    ls2 = la.solve_triangular(r, q.T @ yk)

    # plot points and approximations
    domain = np.linspace(min(xk), max(xk), 1000)
    plt.scatter(xk, yk, label="Original", c="k")
    plt.plot(domain, np.polyval(ls1, domain), label="Normal Equations", c="r")
    plt.plot(domain, np.polyval(ls2, domain), label="Triangular Equations", c="g")

    # set plot parameters
    plt.xlabel("Domain")
    plt.ylabel("Codomain")
    plt.title(f"Problem 5: n={n}")
    plt.ylim(min(yk)*.9, max(yk)*1.1)
    plt.legend()
    plt.show()

    # return condition numbers
    return npla.norm(A@ls1 - yk), npla.norm(A@ls2 - yk)



def integral_error():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. 
    """

    # initialize local variables
    x = sy.symbols("x")
    error = []
    nlist = []

    for n in range(5, 51, 5):
        # calculate symbolically
        Isym = float(sy.integrate(x**n * sy.exp(x-1), (x, 0, 1)))

        # calculate numerically
        Isub = (-1)**n * (sy.subfactorial(n) - sy.factorial(n) / np.e)

        # update lists
        error.append(np.abs(Isym - Isub))
        nlist.append(n)

    # set plot parameters
    plt.plot(nlist, error, label="error", c='k')
    plt.yscale("log")
    plt.xlabel("Values of N")
    plt.ylabel("Error")
    plt.title("Problem 6")
    plt.legend()
    plt.show()
