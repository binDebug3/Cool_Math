import cvxpy as cp
import numpy as np



def piano_shipping():
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



def optimize_nutrition():
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