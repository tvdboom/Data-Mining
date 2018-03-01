"""
-------------------------------------------------------------------------------
                              AiDM - Assignaturenment 3
                   Implementation of LSH using minhashing
                       Lennart van Sluijs & Marco TvdB
-------------------------------------------------------------------------------
                                FUNCTIONS
-------------------------------------------------------------------------------
"""

# import packages
import numpy as np
import scipy.sparse
import time

def get_M(x, per):
    """
    Calculates the matrix M for the data for hash functions represented by
    a list of permutations.
    
    INPUT PARAMETERS:
    x - binary values matrix
    per - permutations to use    
    
    OUTPUT PARAMETERS:
    M - all M values for all hash functions for all users
    """

    # allocate memory for M
    M = np.zeros((per.shape[0], x.shape[1]))
    
    # get x in row base for permutations
    x_csr = x.tocsr()
    
    # get values
    for p in range(per.shape[0]):
        
        print 'Signature of permutation: (',p+1,'/',per.shape[0],')'
        # permute matrix using permutation
        x_per = x_csr[per[p],:]
        x_per = x_per.tocsc() # back to column base
        
        # loop over all columns
        M[p,:] = x_per.argmax(axis = 0)
                    
    return M

def get_sparse_matrix(x):
    """
    Creates a matrix representation of data x.
    
    INPUT PARAMETERS:
    x - array with 2 columns containing list of pairs in data set.
    
    OUTPUT PARAMETERS:
    x_matrix - matrix representation of x
    n_users - number of users in data set
    n_movies - number of movies in data set
    """
    
    # allocate memory for x
    n_users = np.max(x[:,0]) + 1 #since lowest ID has value 0
    n_movies = np.max(x[:,1]) + 1 #since lowest ID has value 0
    zeros = np.zeros(x.shape[0]) + 1 # fill with zeros
    x_matrix = scipy.sparse.csc_matrix((zeros.astype(int),(x[:,1], x[:,0])),
                                       shape=(n_movies, n_users))
    
    return x_matrix, n_users, n_movies

def get_jacobysim(x1, x2):
    """
    Calculates the Jacoby similarity between two objects.
    
    INPUT PARAMETERS:
    x1 - binary-values array
    x2 - binary-values array
    
    OUTPUT PARAMETER:
    jacobysim - jacoby similarity of x1 and x2
    """
    
    # get javoby similarity
    ind1 = np.where(x1 == 1)[0]
    ind2 = np.where(x2 == 1)[0]
    n_both = float(len(np.intersect1d(ind1, ind2)))
    n_total = float(len(np.union1d(ind1, ind2)))
    jacobysim = n_both/n_total
    
    return jacobysim
    
def get_all_permutations(x):
    """
    Create a list of all permutations of array x.
    
    INPUT PARAMETERS:
    x - array
    
    OUTPUT PARAMETERS:
    permutations- correpsonding permuted array
    """
    
    # get permuations
    permutations = np.empty((0,2), int)
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            permutations = np.vstack((permutations, [x[i],x[j]]))
            
    return permutations
    
def get_candidates(M, b):
    """
    Calculate the initial candidates using b bands of r rows.
    
    INPUT PARAMETERS:
    M - matrix containing M values
    b - number of bands
    
    OUTPUT PARAMETERS:
    init_candidates - list of the initial candidates to check
    """

    # get bands
    bands = np.array_split(M, b, axis = 0)

    # hash snippets of each band
    candidates = np.empty((0,2), int)
    for band_key, band in enumerate(bands):
        dictionary = {} # create initally an empty dictionary for this band
        
        print 'Comparing band: (',band_key+1,'/',b,')'
        for user_key, column in enumerate(band.T):
            dictionary_id = tuple(column) # create a bucket ID
            
            # if not yet in dictionary yet
            if dictionary_id not in dictionary:
                dictionary[dictionary_id] = tuple([user_key]) # also save user_key
            else: # add to user_key dictionary and save as candidates
                dictionary[dictionary_id] = dictionary[dictionary_id] + tuple([user_key])
                user_keys = np.array(dictionary[dictionary_id])
                new_candidates = get_all_permutations(user_keys)
                candidates = np.vstack((candidates, new_candidates))        
                
    # get rid of all duplicates
    candidates = np.unique(candidates, axis = 0)
    
    return candidates
