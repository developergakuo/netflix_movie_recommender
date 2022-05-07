from termios import B110
import numpy as np
from scipy.sparse import csr_matrix, spdiags
import multiprocessing
from itertools import combinations
import random

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 1 Read the movie ratings files into a sparse matrix 
'''
NUMBER_OF_FILES_TO_READ = 0
movie_text_files = ['netflix_dataset/combined_data_{}.txt'.format(i) for i in  range(0,NUMBER_OF_FILES_TO_READ + 1)]

# the movie_ids_list, users_ids_list, and users_ratings_list have the same length and their contents correspond 
movie_ids_list = [] # a list of movie ids 
users_ids_list = [] # a list of lists for user ids, each sublist holds users who rated a movie
users_ratings_list = [] # a list of lists for user ratings, each sublists holds ratings for one movie
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
            movie_ids_list.append(int(movieId))

            if count > 1:
                # appened the user_ids and  their ratings sublists for the previous movie
                # to the users_ids_list and users_ratings_list
                users_ids_list.append(sub_users_ids_list)
                users_ratings_list.append(sub_users_ratings_list)

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
    users_ids_list.append(sub_users_ids_list)
    users_ratings_list.append(sub_users_ratings_list)

    # dispose unused objects 
    del sub_users_ids_list
    del sub_users_ratings_list

print('actual ratings dims: ', len(movie_ids_list),len(users_ids_list) , len(users_ratings_list))

# Create a set of all the unique users
unique_users = set()
for lst in users_ids_list:
    unique_users.update(lst)

print('unique_users: ', len(unique_users))

index = 0
for lst in users_ids_list:
    #find all the users who did not rate a movie
    unrated_users = list(unique_users - set(lst))
        
    # update the users for each movie to include all users who have not yet rated the movie
    users_ids_list[index] = lst + unrated_users

    # fill out the entries of users who did not rate a movie with zero 
    users_ratings_list[index] = users_ratings_list[index] + np.zeros_like(unrated_users).tolist()
    index += 1

# read users_ratings_list into a sparse matrix 
users_ratings_list = np.array(users_ratings_list, dtype=np.int32)

# movies as rows, users as columns 
short_wide_sparse_matrix = csr_matrix(users_ratings_list)
print("short_wide_sparse_matrix shape: ",  short_wide_sparse_matrix.shape)

# movies as columns, users as rows 
tall_thin_sparse_matrix = short_wide_sparse_matrix.transpose()
print("tall_thin_sparse_matrix shape: ",  tall_thin_sparse_matrix.shape)

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 2: DimSum
'''
m,n = tall_thin_sparse_matrix.shape
B = np.zeros((n,n))

gamma = 10000000 # sampling parameter 
rows = [tall_thin_sparse_matrix.getrow(i) for i in range(0,m) ]

# compute the l2-norms of the columns
columns = [tall_thin_sparse_matrix.getcol(i) for i in range(0,n)]
column_l2_norms = [np.sqrt(column.power(2).sum()) for column in columns]
D  = spdiags(column_l2_norms, 0, len(column_l2_norms), len(column_l2_norms))

del columns

# mapper 
key_value_pairs = {} # output values of the for j,k -> aij * aik
for rowi in rows:
    for j,k in combinations(range(0,n), 2): 
        sampling_probability = gamma * (1 / (column_l2_norms[j]* column_l2_norms[k]))
        sampling_probability = sampling_probability if (sampling_probability < 1) else 1

        if random.random() < sampling_probability:
            output = rowi[0,j] * rowi[0,k] # each row of the sparse matrx is indexed as (0, i)
            if output > 0: # only output values greater than 0
                if (j,k) in key_value_pairs:
                    values = key_value_pairs[(j,k)]
                    values.append(output)
                    key_value_pairs[(j,k)] = values
                else:
                    key_value_pairs[(j,k)] = [output]

#reducer 
for (j,k), values in key_value_pairs.items():
    denom = (column_l2_norms[j] * column_l2_norms[k])
    sampling_probability = gamma * (1 / denom)
    if sampling_probability > 1:
        B[j,k] = sum(values)/denom # normalise where probability was greater than 1

    else:
         B[j,k] =  sum(values)/gamma # estimate the expectation where probability was less than 1

print('B', B)

A_transpose_A_estimate = D @ B @ D
print('A_transpose_A_estimate', A_transpose_A_estimate)

A_transpose_A = tall_thin_sparse_matrix.transpose() @ tall_thin_sparse_matrix
print('A_transpose_A', A_transpose_A)

difference_matrix = csr_matrix(A_transpose_A - A_transpose_A_estimate )
average_min_squared_error = difference_matrix.power(2).sum()/(n*n)

print('average_min_squared_error', average_min_squared_error)

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 3: Gradient Descent Computation
'''

''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 4: Accuracy calculation
'''