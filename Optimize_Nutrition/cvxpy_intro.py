# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> Dallin Stewart
<Class> ACME 002
<Date> 3/2/23
"""
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:
    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x, y, z = cp.Variable(), cp.Variable(), cp.Variable()

    # set constraints
    constraints = [x + 2*y <= 3,
                   y - 4*z <= 1,
                   2*x + 10 * y + 3*z >= 12,
                   x >= 0, y >= 0, z >= 0]

    # compute objective and solution
    objective = cp.Minimize(2*x + y + 3*z)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimizer = np.array([x.value, y.value, z.value])

    return optimizer, problem.value


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||x||_1
        subject to  Ax = b
    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)
    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set variable list
    # n = A.shape[1]
    x = cp.Variable(len(A[1]))

    # set constraints
    constraints = [A @ x == b]

    # set objective
    objective = cp.Minimize(cp.norm(x, 1))

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value, problem.value


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    p = cp.Variable(6, nonneg=True)

    # set objective
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ p)

    # set constraints
    A = np.array([[1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1]])
    P = np.eye(6)
    b = np.array([7, 2, 4, 5, 8])
    constraints = [A @ p == b, P @ p >= 0]

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()

    return p.value, solution


# Problem 4
def prob4():
    """Find the minimizer and minimum of
    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set Q and r based on the function g
    Q = np.array([[3, 2, 1],
                  [2, 4, 2],
                  [1, 2, 3]])
    r = np.array([3, 0, 1])

    # set variables and constraints
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))

    # solve problem and return solution
    solution = prob.solve()
    return x.value, solution


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # set variables
    n = A.shape[1]
    x = cp.Variable(n, nonneg=True)

    # set objective
    objective = cp.Minimize(cp.norm(A@x - b, 2))

    # set constraints
    P = np.eye(n)
    constraints = [cp.sum(x) == 1, P @ x >= 0]

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()

    return x.value, solution


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # load food matrix
    food = np.load("food.npy", allow_pickle=True)
    n = food.shape[0]
    # multiply nutrition by the number of servings
    food[:,2:] = food[:,2:] * food[:, 1][:, np.newaxis]
    # split table into each category
    price, servings, calories, fat, sugar, calcium, fiber, protein = [food[:,j] for j in range(food.shape[1])]

    # set objective
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(price @ x)

    # set constraints
    P = np.eye(n)
    constraints = [calories @ x <= 2000,
                   fat @ x <= 65,
                   sugar @ x <= 50,
                   calcium @ x >= 1000,
                   fiber @ x >= 25,
                   protein @ x >= 46,
                   P @ x >= 0]

    # Assemble the problem and then solve it
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()

    return x.value, solution


if __name__ == "__main__":
    pass
    # # test problem 1
    # print("\n Problem 1")
    # print(prob1())

    # # test problem 2
    # A = np.array([[1, 2, 1, 1],
    #               [0, 3, -2, -1]])
    # b = np.array([7, 4])
    # print(l1Min(A, b))

    # # test problem 3
    # print("\n Problem 3")
    # print(prob3())
    #
    # # test problem 4
    # print("\n Problem 4")
    # print(prob4())
    #
    # # test problem 5
    # print("\n Problem 5")
    # Atest = np.array([[1, 2, 1, 1],
    #                   [0, 3, -2, -1]])
    # btest = np.array([7, 4]).T
    # print(prob5(Atest, btest))
    #
    # # test problem 6
    # print("\n Problem 6")
    # result = prob6()
    # optimum = result[0]
    # print(optimum)
    # a = result[1]
    # for i in range(a.size):
    #     if a[i] < 0.0001:
    #         a[i] = 0
    # print(a)