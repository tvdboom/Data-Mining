"""
-------------------------------------------------------------------------------
                              AiDM - Assignment 3
                   Implementation of LSH using minhashing
                       Lennart van Sluijs & Marco TvdB
-------------------------------------------------------------------------------
                                     MAIN
-------------------------------------------------------------------------------
"""

# import packages
import sys
import numpy as np
from lsh_functions import * # contains functions used to perform lsh
import scipy.sparse
import time

print '----------------------------------------------------------------'
print '                   LSH Assignment 3 AidM                        '
print '               Lennart van Sluijs & Marco TvdB'
print '----------------------------------------------------------------'

# parameters to use
n_hashfunctions = 100
threshold = 0.5 # search for > 50% - similarity pairs
b = 10
r = int(n_hashfunctions/b)
print 'Parameters used: '
print '- # hashfunctions:',n_hashfunctions
print '- # threshold:',threshold
print '- # bands:',b
print '- # rows per band:',r
print ''

# get arguments from command line
randomseed, fname = (int(sys.argv[1]), sys.argv[2])

# time when algorithm starts
time_start = time.time()

# load the data
print 'Loading data.'
user_movie_pairs = np.load(fname) -1 # set 1st user/movie to 0
print 'Starting LSH.'
print 'Creating matrix.'

# use a matrix representation of the data
user_movie_matrix, n_users, n_movies = get_sparse_matrix(user_movie_pairs)

# initalize hash functions or permutations
per = np.zeros((n_hashfunctions, n_movies), dtype = 'int')
print 'Initialzing hash functions.'
for h in range(n_hashfunctions):
    np.random.seed(randomseed + h)
    per[h,:] = np.random.permutation(n_movies)

print 'Get M.'
M = get_M(user_movie_matrix, per)

# get inital candidates
print 'Get candidates.'
candidates = get_candidates(M, b)
print ''
print 'Candidate user pairs to check: '
print candidates

print ''
print 'Checking all candidate user pairs: '
similar_users = np.empty((0,2), int)
for i in range(candidates.shape[0]):
    
    # get candidate ids
    id1 = candidates[i,0]
    id2 = candidates[i,1]

    # calculate jacoby similarity
    jacobysim = get_jacobysim(user_movie_matrix[:,id1].toarray(),
                              user_movie_matrix[:,id2].toarray())

    if jacobysim > threshold:
        similar_users = np.vstack((similar_users, [id1+1, id2+1]))
        np.savetxt('results.txt', similar_users.astype(int), fmt='%i',
                   delimiter = ',')
        print ''
        print [id1+1,id2+1]
        print 'Jacoby similarity:',jacobysim
        print 'Similar users! Saved results.'

print ''       
print 'Finished.'
print 'Total time:',time.time()-time_start,'s'
print 'Total candidates found:',similar_users.shape[0]
print 'Candidates found/minute',float(similar_users.shape[0])/(float(time.time()-time_start)/60.)
