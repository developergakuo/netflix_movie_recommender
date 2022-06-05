from termios import B110
import numpy as np
from scipy.sparse import csr_matrix, spdiags, lil_matrix
from scipy.sparse.linalg import svds

import multiprocessing
from itertools import combinations, permutations
import random

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 1 Read the movie ratings files into a sparse matrix 
'''
NUMBER_OF_FILES_TO_READ = 0
movie_text_files = ['netflix_dataset/combined_data_{}.txt'.format(i) for i in  range(0,NUMBER_OF_FILES_TO_READ + 1)]

# the movie_ids_list, users_ids_list, and users_ratings_list have the same length and their contents correspond 
test_movie_ids_list = [] # a list of movie ids 
test_users_ids_list = [] # a list of lists for user ids, each sublist holds users who rated a movie
test_users_ratings_list = [] # a list of lists for user ratings, each sublists holds ratings for one movie


train_movie_ids_list = [] # a list of movie ids 
train_users_ids_list = [] # a list of lists for user ids, each sublist holds users who rated a movie
train_users_ratings_list = [] # a list of lists for user ratings, each sublists holds ratings for one movie

for file_path in movie_text_files:
    file = open(file_path, 'r')
    Lines = file.readlines()

    count = 0

    sub_users_ids_list = [] # a list to hold the user ids for each movie
    sub_users_ratings_list = [] # a list to hold each user ratings for each movie
    for line in Lines:
        count += 1
        line = line.strip()
        if ':' in line:
            movieId = line[:-1]
            test_movie_ids_list.append(int(movieId))

            if count > 1:
                # appened the user_ids and  their ratings sublists for the previous movie
                # to the users_ids_list and users_ratings_list
                test_users_ids_list.append(sub_users_ids_list)
                test_users_ratings_list.append(sub_users_ratings_list)

                #clear the user and ratings sublists for the next movie
                sub_users_ids_list = []
                sub_users_ratings_list = []
        else:
            line = line.strip().split(',')
            user_id = line[0]
            user_rating = line[1]

            sub_users_ids_list.append(int(user_id))
            sub_users_ratings_list.append(int(user_rating))

    # make sure to add the user ids and ratings of the last movie to the users_ids_list, and users_ratings_list
    test_users_ids_list.append(sub_users_ids_list)
    test_users_ratings_list.append(sub_users_ratings_list)

    # dispose unused objects 
    del sub_users_ids_list
    del sub_users_ratings_list

print('Number of movies rated: ', len(test_movie_ids_list))

# Create a set of all the unique users
unique_users = set()
for lst in test_users_ids_list:
    unique_users.update(lst)

print('unique_users: ', len(unique_users))

index = 0
for lst in test_users_ids_list:
    #find all the users who did not rate a movie
    unrated_users = list(unique_users - set(lst))
        
    # update the users for each movie to include all users who have not yet rated the movie
    test_users_ids_list[index] = lst + unrated_users

    # fill out the entries of users who did not rate a movie with zero 
    test_users_ratings_list[index] = test_users_ratings_list[index] + np.zeros_like(unrated_users).tolist()
    index += 1

# read users_ratings_list into a sparse matrix 
test_users_ratings_list = np.array(test_users_ratings_list, dtype=np.int8)

# movies as rows, users as columns 
short_fat_sparse_matrix = csr_matrix(test_users_ratings_list, dtype=np.int8)
print("short_fat_sparse_matrix shape: ",  short_fat_sparse_matrix.shape)

# movies as columns, users as rows 
tall_skinny_sparse_matrix = short_fat_sparse_matrix.transpose()
print("tall_skinny_sparse_matrix shape: ",  tall_skinny_sparse_matrix.shape)

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 2: DimSum
'''
m,n = tall_skinny_sparse_matrix.shape
B = lil_matrix((n,n))

# sampling parameter, the higher the sampling parameter the higher the sampled entries of the  tall_skinny_sparse_matrix
# higher sampling rate leads to higher accuracy of the estimated ATA matrix
gamma = 1000000  
# compute the l2-norms of the columns

def get_l2_norms():
    column_l2_norms = []
    for i in range(0,n):
       column_l2_norms.append(np.sqrt(tall_skinny_sparse_matrix.getcol(i).power(2).sum())) 
    return column_l2_norms

column_l2_norms = get_l2_norms()


D  = spdiags(column_l2_norms, 0, n, n).tocsr()


# mapper - read in rows and output each pair (aij*aik) with a probability (where i is the row, and j & k are the columns)
key_value_pairs = {} # output values of the for j,k -> aij * aik
for rowi in range(0,m):
    for i in range(0,n): # pair each column j with each column k 
        for j in range(0,n):
            sampling_probability = gamma * (1 / (column_l2_norms[i]* column_l2_norms[j]))
            sampling_probability = sampling_probability if (sampling_probability < 1) else 1

            if random.random() < sampling_probability:
                output = tall_skinny_sparse_matrix.getrow(rowi)[0,i] * tall_skinny_sparse_matrix.getrow(rowi)[0,j] # each row of the scipy sparse matrx is indexed as (0, i)
                if output > 0: # only output values greater than 0
                    if (i,j) in key_value_pairs:
                        values = key_value_pairs[(i,j)]
                        values.append(output)
                        key_value_pairs[(i,j)] = values
                    else:
                        key_value_pairs[(i,j)] = [output]

#reducer - add the entries of the key_value_pairs 
# - normalise where probability was greater than 1
# - estimate the expectation where probability was less than 1

for (i,j), values in key_value_pairs.items():
    denom = (column_l2_norms[i] * column_l2_norms[j])
    sampling_probability = gamma * (1 / denom)
    if sampling_probability > 1:
        B[i,j] = sum(values)/denom # normalise where probability was greater than 1
    else:
         B[i,j] =  sum(values)/gamma # estimate the expectation where probability was less than 1


B = B.tocsr()
A_transpose_A_estimate = D @ B @ D

A_transpose_A = tall_skinny_sparse_matrix.transpose() @ tall_skinny_sparse_matrix

difference_matrix = csr_matrix(A_transpose_A - A_transpose_A_estimate )
average_min_squared_error = difference_matrix.power(2).sum()/(n*n)

print('average_min_squared_error: ', average_min_squared_error)
''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 3: Gradient Descent Computation
'''
m,n = short_fat_sparse_matrix.shape

# `k` must be an integer satisfying `0 < k < min(A.shape)`.
j = m-1
Q, s, vh = svds(short_fat_sparse_matrix, k=j)
s = np.diag(s)
Ph = s @ vh

print(s.shape, vh.shape)
print( "matrix Q - shape: ", Q.shape)
print( "matrix Ph - shape: ", Ph.shape)


def prefict_rating(users_ratings_list, user_vector):
    return np.dot(users_ratings_list, user_vector)
# users_ratings_list[i,x] is approximated by (Q[i:].Transpose() * Ph[:x]) for the known etries of the users_ratings_list matrix
def SGD():
    pass

def BGD():
    for i in range(0,m): # movies in rows
        for u in range(0,n): # users in columns
            pass








''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 4: Accuracy calculation
'''


