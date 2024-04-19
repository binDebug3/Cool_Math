import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse


class NMFRecommender:

    def __init__(self,random_state=15,rank=2,maxiter=200,tol=1e-3):
        """
        Save the parameter values as attributes.
        """
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
  

    def _initialize_matrices(self, m, n):
        """
        Initialize the W and H matrices.
        
        Parameters:
            m (int): the number of rows
            n (int): the number of columns
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        # initialize matrices
        np.random.seed(self.random_state)
        W = np.random.rand(m, self.rank)
        H = np.random.rand(self.rank, n)
        return W, H


    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm according to the 
        Frobenius norm.
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        """
        # compute frobenius norm
        return np.linalg.norm(V - W @ H, ord='fro')


    def _update_matrices(self, V, W, H):
        """
        The multiplicative update step to update W and H
        Return the new W and H (in that order).
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            New W ((m,k) array)
            New H ((k,n) array)
        """
        # update matrices
        H = H * (W.T @ V) / (W.T @ W @ H)
        W = W * (V @ H.T) / (W @ H @ H.T)
        return W, H


    def fit(self, V):
        """
        Fits W and H weight matrices according to the multiplicative 
        update algorithm. Save W and H as attributes and return them.
        
        Parameters:
            V ((m,n) array): the array to decompose
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        # initialize matrices
        m, n = V.shape
        W, H = self._initialize_matrices(m, n)
        
        # update matrices
        for i in range(self.maxiter):
            W, H = self._update_matrices(V, W, H)
            loss = self._compute_loss(V, W, H)
            if loss < self.tol:
                break
        
        # save as attributes
        self.W = W
        self.H = H
        return W, H


    def reconstruct(self):
        """
        Reconstruct and return the decomposed V matrix for comparison against 
        the original V matrix. Use the W and H saved as attrubutes.
        
        Returns:
            V ((m,n) array): the reconstruced version of the original data
        """
        # reconstruct V
        return self.W @ self.H


def run_nmf(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Returns:
        W ((m,k) array)
        H ((k,n) array)
        The number of people with higher component 2 than component 1 scores
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
                  
    # run NMF recommender
    nmf = NMFRecommender(rank=rank)
    W, H = nmf.fit(V)

    # count number of people with higher component 2 than component 1 scores
    return W, H, np.sum(H[1,:] > H[0,:])


def compute_rank(filename='artist_user.csv', set_rank=None):
    """
    Read in the file `artist_user.csv` as a Pandas dataframe. Find the optimal
    value to use as the rank as described in the lab pdf. Return the rank and the reconstructed matrix V.
    
    Returns:
        rank (int): the optimal rank
        V ((m,n) array): the reconstructed version of the data
    """
    import os
    from sklearn.decomposition import NMF

    # read in data
    X = pd.read_csv(filename, index_col=0)
    # X = df.values
    rstate = 0

    # find benchmark rank
    benchmark = np.linalg.norm(X, ord='fro') * 0.0001
    best_rank = 0
    best_v = X

    if set_rank:
        scan = range(set_rank, set_rank+1)
    else:
        scan = range(10, 16)
    for r in scan:
        # run NMF recommender
        model = NMF(n_components=r, init='random', random_state=rstate, max_iter=1000)
        W = model.fit_transform(X)
        H = model.components_
        V_recon = W @ H
        print("Rank:", r)

        # compute error
        if np.sqrt(mse(X, V_recon)) < benchmark:
            best_rank = r
            best_v = V_recon
            break

    return best_rank, best_v


def discover_weekly(userid, V):
    """
    Create the recommended weekly 30 list for a given user.
    
    Parameters:
        userid (int): which user to do the process for
        V ((m,n) array): the reconstructed array
        
    Returns:
        recom (list): a list of strings that contains the names of the recommended artists
    """
    df = pd.read_csv('artist_user.csv', index_col=0)
    df.rename(columns=pd.read_csv('artists.csv').astype(str).set_index('id').to_dict()['name'], inplace=True)
    users = df.loc[userid, :].reset_index()
    users.rename(columns={'index':'artist_id', userid:'plays'}, inplace=True)

    # get recommended artists
    users['recoms'] = V[userid-2]
    users = users[users['plays'] == 0]
    users.sort_values(by='recoms', ascending=False, inplace=True)

    # return top 30
    return users.head(30)['artist_id'].tolist()
    


if __name__ == "__main__":
    W, H, num = run_nmf()
    print("W:\n", W)
    print("H:\n", H)
    print("Number of people with higher component 2 than component 1 scores:", num)

    rank, V = compute_rank(set_rank=13)
    print("Rank:", rank)

    print("The recommended weekly 30 list for a given user: ")
    print(discover_weekly(2, V))