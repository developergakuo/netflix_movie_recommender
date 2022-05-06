import numpy as np
from scipy.sparse import csr_matrix
import multiprocessing

'''
TASK 1 Read the movie ratings files into a sparse matrix 
'''
NUMBER_OF_FILES_TO_READ = 1
movie_text_files = ['netflix_dataset/combined_data_{}.txt'.format(i) for i in  range(1,NUMBER_OF_FILES_TO_READ + 1)]

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
        
    # update the users for each movie to 
    users_ids_list[index] = users_ids_list[index] + unrated_users

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
'''
TASK 2: DimSum
'''

'''
TASK 3: Gradient Descent Computation
'''

'''
TASK 4: Accuracy calculation
'''