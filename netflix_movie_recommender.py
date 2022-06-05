from calendar import EPOCH
from distutils.log import error
from enum import unique
from functools import reduce
import re
from termios import B110
import numpy as np
from scipy.sparse import csr_matrix, spdiags, lil_matrix, coo_matrix
from scipy.sparse.linalg import svds

import multiprocessing
from itertools import combinations, permutations
import random
import multiprocessing as mp
import json



''' ----------------------------------------------------------------------------------------------------------------------------------------------------
TASK 1 Read the movie ratings files into a sparse matrix 
'''
NUMBER_OF_FILES_TO_READ = 3
movie_text_files = ['netflix_dataset/combined_data{}_1.txt'.format(i) for i in  range(1,NUMBER_OF_FILES_TO_READ + 1)]

probe_file_name = 'netflix_dataset/probe2.txt'

def read_movie_ratings_file(file_path, queue= None, in_process = True):
    """reads movie ratings files into a sparse array

        If the argument `in_process` is true, the argument queue must be given
        Parameters
        ----------
        file_path : str, required
            movie ratings file
   
        queue : Queue, optional
            used to share returned value test and train tuples in the multiprocessing implementation
        in_process : Bool, optional
            if True, the function puts values in queue 

    """
    print("Reading movie text file: {} ".format(file_path))
    # the movie_ids_list, users_ids_list, and users_ratings_list have the same length and their contents correspond for both test and train
    train_movie_ids_list = [] # a list of movie ids 
    train_users_ids_list = [] # a list for user ids
    train_users_ratings_list = [] # a list of user ratings

    with open(file_path) as infile:
        for line in infile:
            line = line.strip()
            if ':' in line:
                movieId = line[:-1]
                
            else:
                line = line.split(',')
                user_id = int(line[0])
                user_rating = line[1]

                train_movie_ids_list.append(movieId)
                train_users_ids_list.append(user_id)
                train_users_ratings_list.append(user_rating)
        if in_process:
            ret_value = queue.get()
            ret_value.append([train_movie_ids_list,train_users_ids_list, train_users_ratings_list])
            queue.put(ret_value)
        else:
            return [train_movie_ids_list,train_users_ids_list, train_users_ratings_list]

def read_probe_file(file_path):
    """reads probe file into a dict ids -> <user ids list> 

        Parameters
        ----------
        file_path : str, required
            probe file path
    """
    print("Reading probe test")

    movie_id_user_id_tuples = [] #dict of movie ids and users in test set 
    file = open(file_path, 'r')
    Lines = file.readlines()

    for line in Lines:
        line = line.strip()
        if ':' in line:
            movieId = line[:-1]
        else:
            line = line.strip()
            user_id = int(line)
            movie_id_user_id_tuples.append([user_id, movieId])
    return movie_id_user_id_tuples

def prune_matrix_test_ratings(m, test_users, test_movies ):
    """Prunes the test matrix from the test matrix 
       Returns a list of test ratings

        Parameters
        ----------
        m : Matrix, required
            Test matrix
        test_users : List, required
            list of test users
        test_movies : List, required
            List of test movies
    """
    print("Start - Prunning the test matrix from the test matrix")

    test_ratings = m[test_users, test_movies].A1
    print("test_ratings", test_ratings.shape)
    test_matrix = csr_matrix((test_ratings, (test_users, test_movies)),shape=m.shape, dtype=np.int64)
    m[test_users, test_movies] = 0
    #m.eliminate_zeros()
    print("Finished - Prunning the test matrix from the test matrix")
    return test_matrix

def create_movie_and_user_dicts(movie_ratings,test_movie_id_user_id_tuples):
        """creates two dicts mapping user ids and movie ids to indexes in the sparse matrix

        Parameters
        ----------
        movie_ratings : list, required
            A list of tuples containging movie_ids, user_ids, and user_ratings
        """
        unique_users = set()
        unique_movies = set()

        for movie_ids_list,users_ids_list, _ in movie_ratings:
            unique_movies.update(movie_ids_list)
            unique_users.update(users_ids_list)

        test_movies = []
        test_users = []
        for userId, movieId in test_movie_id_user_id_tuples:
            test_movies.append(movieId)
            test_users.append(userId)

        
        
        available_test_users = unique_users.intersection(set(test_users))
        available_test_movies = unique_movies.intersection(set(test_movies))
      
        print("available_test_users", available_test_users)
        print("available_test_movies", available_test_movies)


        n = len(list(unique_movies))
        m = len(list(unique_users))

        print("unique movies",n)
        print("unique users", m)

        user_ids_dict = {}
        for v,k in enumerate(unique_users):
                user_ids_dict[k] = v

        movie_ids_dict = {}
        for v,k in enumerate(unique_movies):
            movie_ids_dict[k] = v

        available_test_tuples = []
        for userId, movieId in test_movie_id_user_id_tuples:
            if movieId in available_test_movies:
                if userId in available_test_users:
                    available_test_tuples.append([user_ids_dict[userId],movie_ids_dict[movieId]])

         
        test_movies = list(map(lambda x: x[1], available_test_tuples))
        test_users = list(map(lambda x: x[0], available_test_tuples))

        return [movie_ids_dict, user_ids_dict, n, m,test_users, test_movies]

def compute_tall_skinny_sparse_matrix(movie_ratings, movie_ids_dict, user_ids_dict, n, m):
        """Creates a sparse matrix of size (m,n)

        Parameters
        ----------
        movie_ratings : list, required
            A list of tuples containging movie_ids, user_ids, and user_ratings

        movie_ids_dict : dict, required
            A dict that maps a movie id to an index in the sparse matrix

        user_ids_dict : dict, required
            A dict that maps a user id to an index in the sparse matrix

        n : int, required
            The number of columns in the sparse matrix
        m : int, required
            The number of rows in the sparse matrix
          
        """
        print("Computing the compute_tall_skinny_sparse_matrix")

        movie_ids_list,users_ids_list, users_ratings_list = [], [], []
        for m_list,u_list, r_list in movie_ratings:
            m_list = list(map(lambda x:  movie_ids_dict[x], m_list))
            u_list = list(map(lambda x:  user_ids_dict[x], u_list))

            movie_ids_list = movie_ids_list + m_list
            users_ids_list = users_ids_list + u_list
            users_ratings_list = users_ratings_list + r_list

        tall_skinny_sparse_matrix = csr_matrix((users_ratings_list, (users_ids_list, movie_ids_list)),shape=(m,n), dtype=np.int64)
        return tall_skinny_sparse_matrix

ret_value = []
if __name__ == '__main__':

        # read in probe file into a dict
        test_movie_id_user_id_tuples = read_probe_file(probe_file_name)

        print( "number of ratings in test-probe: ", len(test_movie_id_user_id_tuples))

        """         # Read in movie rating files in parallel into test and train sets
        processes = []
        queue = mp.Queue()
        queue.put(ret_value)
        for file in movie_text_files:
            print("reading ratings file: {}".format(file))
            
            p = mp.Process(target=read_movie_ratings_file, args=(file,queue))
            p.start()
            processes.append(p)
        for p in processes:
            p.join() """

        train_movie_ratings = []
        for filepath in movie_text_files:
             train_movie_ids_list,train_users_ids_list, train_users_ratings_list = read_movie_ratings_file(filepath, in_process=False)
             train_movie_ratings = train_movie_ratings + [[train_movie_ids_list,train_users_ids_list, train_users_ratings_list]]

        # map user ids and movie ids to matrix indexes
        # without this, users and movies might map to indexes beyond the size of the matrix in the m dimension 
        movie_ids_dict, user_ids_dict, n, m,test_users, test_movies = create_movie_and_user_dicts(train_movie_ratings, test_movie_id_user_id_tuples)

       
        '''
        train sets and test sets
        '''
        # movies as columns, users as rows 
        train_tall_skinny_sparse_matrix = compute_tall_skinny_sparse_matrix(train_movie_ratings, movie_ids_dict, user_ids_dict, n, m)
        print("train_tall_skinny_sparse_matrix shape: ",  train_tall_skinny_sparse_matrix.shape)
        print("Number of ratings in  matrix BEFORE prune",len(train_tall_skinny_sparse_matrix.nonzero()[1]))

        #eliminate the test movie etries and produce a test matatrix
        test_tall_skinny_sparse_matrix = prune_matrix_test_ratings(train_tall_skinny_sparse_matrix, test_users, test_movies)
        print("test_tall_skinny_sparse_matrix shape: ",  test_tall_skinny_sparse_matrix.shape)
        print("test_tall_skinny_sparse_matrix non-zero: ",  len(test_tall_skinny_sparse_matrix.nonzero()[0]))

        print("train_tall_skinny_sparse_matrix shape: ",  train_tall_skinny_sparse_matrix.shape)
        print("Number of ratings in train_tall_skinny_sparse_matrix AFTER prune",len(train_tall_skinny_sparse_matrix.nonzero()[0]))


        # movies as rows, users as columns 
        train_short_fat_sparse_matrix = train_tall_skinny_sparse_matrix.transpose()
        print("train_short_fat_sparse_matrix shape: ",  train_short_fat_sparse_matrix.shape)

        # movies as columns, users as rows 
        test_short_fat_sparse_matrix = test_tall_skinny_sparse_matrix.transpose()
        print("test_short_fat_sparse_matrix shape: ",  test_short_fat_sparse_matrix.shape)


        ''' ----------------------------------------------------------------------------------------------------------------------------------------------------
        TASK 2: DimSum
        '''

        def get_l2_norms(A):
            """"Computes the l2_norms for all the columns of a matrix A
                """

            print("Calculating  l2_Norms with function get_l2_norms")
            _,n = A.shape
            column_l2_norms = []
            for i in range(0,n):
                column_l2_norms.append(np.sqrt(A.getcol(i).power(2).sum())) 
            return column_l2_norms

        def run_dimsum():
            m,n = train_tall_skinny_sparse_matrix.shape
            # compute the l2-norms of the columns
        

            column_l2_norms = get_l2_norms(train_tall_skinny_sparse_matrix.astype(np.int64))
            print("column_l2_norms: ", column_l2_norms)
          
            max_norm = max(column_l2_norms)
            
            max_norm = int(max_norm)
            max_dimsum_gamma = (max_norm * max_norm)
            dimsum_step = int(max_dimsum_gamma/10)
            max_dimsum_gamma = (max_dimsum_gamma + dimsum_step)
            print("max_dimsum_gamma", int(max_dimsum_gamma))

                # create the D matrix from the column_l2_norms
            D  = spdiags(column_l2_norms, 0, n, n).tocsr()
            
            # sampling parameter, the higher the sampling parameter the higher the sampled entries of the  tall_skinny_sparse_matrix
            # higher sampling rate leads to higher accuracy of the estimated ATA matrix
            dimsum_rmse = {}
            for gamma in range(0, max_dimsum_gamma , dimsum_step):

                # Build up the B matrix as a lil_matrix then convert it to a csr_matrix in the end. 
                # Note: 
                # 1.  A dense numpy matrix is faster, but we have memory limitations 
                # 2.  Building up a sparse csr_matrix is costly
                B = lil_matrix((n,n))

                # mapper - read in rows and output each pair (aij*aik) with a probability (where i is the row, and j & k are the columns)
                key_value_pairs = {} # output values of the for j,k -> aij * aik
                for rowi in range(0,m):
                    for i in range(0,n): # pair each column j with each column k 
                        for k in range(0,n):
                            sampling_probability = gamma * (1 / (column_l2_norms[i]* column_l2_norms[k]))
                            sampling_probability = sampling_probability if (sampling_probability < 1) else 1

                            if random.random() < sampling_probability:
                                output = train_tall_skinny_sparse_matrix.getrow(rowi)[0,i] * train_tall_skinny_sparse_matrix.getrow(rowi)[0,k] # each row of the scipy sparse matrx is indexed as (0, i)
                                if output > 0: # only output values greater than 0
                                    if (i,k) in key_value_pairs:
                                        values = key_value_pairs[(i,k)]
                                        values.append(output)
                                        key_value_pairs[(i,k)] = values
                                    else:
                                        key_value_pairs[(i,k)] = [output]

                #reducer - add the entries of the key_value_pairs 
                # - normalise where probability was greater than 1
                # - estimate the expectation where probability was less than 1
                for (i,k), values in key_value_pairs.items():
                    denom = (column_l2_norms[i] * column_l2_norms[k])
                    sampling_probability = gamma * (1 / denom)
                    if sampling_probability > 1:
                        B[i,k] = sum(values)/denom # normalise where probability was greater than 1
                    else:
                        B[i,k] =  sum(values)/gamma # estimate the expectation where probability was less than 1

                # change B matrix to csr_matrix: csr_matrix are more efficient for matrix multiplication
                B = B.tocsr()

                A_transpose_A_estimate = D @ B @ D

                A_transpose_A = train_tall_skinny_sparse_matrix.transpose() @ train_tall_skinny_sparse_matrix

                difference_matrix = csr_matrix(A_transpose_A - A_transpose_A_estimate )
                average_min_squared_error = np.sqrt(difference_matrix.power(2).sum()/(n*n))

                print("running DIMSUM with Gamma = {} - RMSE - {}".format(gamma, average_min_squared_error))


                dimsum_rmse[gamma] = average_min_squared_error

            with open('dimsum_rmse data.json', 'w') as f:       
                    json.dump(dimsum_rmse, f) 
        run_dimsum()
        
        ''' ----------------------------------------------------------------------------------------------------------------------------------------------------
        TASK 4 part 1: Accuracy calculation
        '''

        def calculate_RMSE(Q, Ph, R, non_zero_indexes, n):
            """Calculates the RMSE between the prediction matrix given by (Q @ Ph) and the real matrix R
            The R sparse matrix has 'n' non-zero entries
            """
            cummulitive_error = 0

            # Get the non-zero indexes 

            #use a lil_matrix this allows for efficient indexing 
            R = lil_matrix(R)

            for i,u in non_zero_indexes:
                qi =  Q.getrow(i)
                pu = Ph.getcol(u)

                prediction = qi.dot(pu)[0,0]
                error = R[i,u] - prediction

                cummulitive_error += error * error 
            return np.sqrt(cummulitive_error/n)


        ''' ----------------------------------------------------------------------------------------------------------------------------------------------------
        TASK 3: Gradient Descent Computation
        '''
        m,n = train_short_fat_sparse_matrix.shape
        test_SGD_accuracy = {}
        test_BGD_accuracy = {}
        LAMBDA = 1/10000
        GAMMA_SGD = 0.00001
        GAMMA_BGD = 0.01

        EPOCHS = 30
        for k in range(1,m-1): # `k` must be an integer satisfying `0 < k < min(A.shape)`.
            print("Running SGD and BGD with k = {}".format(k))
            Q, s, vh = svds(train_short_fat_sparse_matrix.asfptype(), k=k)
            s = spdiags(s, 0, k, k).tocsr()
            vh = csr_matrix(vh)
            Ph = s @ vh #ecodes user information
            Q = csr_matrix(Q) #encodes movie information

            print( "matrix Q - shape: ", Q.shape)
            print( "matrix Ph - shape: ", Ph.shape)

            

            # obtain the squared l2_norms of Q rows
            Q_l2_norms = np.array(get_l2_norms(Q.transpose()))
            Q_l2_norms_squared = Q_l2_norms * Q_l2_norms

            # obtain the squared l2_norms of Ph columns
            Ph_l2_norms = np.array(get_l2_norms(Ph))
            Ph_l2_norms_squared = Ph_l2_norms * Ph_l2_norms

        

            # short_fat_sparse_matrix[i,u] is approximated by:
            #  (Q[i:].Transpose() * Ph[:u]) for the known etries of the short_fat_sparse_matrix matrix

            def SGD(i,u, Q,Ph,R):
                    """In one epoch, Changes just a single row in Q and single column in Ph related to a known entry in R[i,u]
                    """
                    actual_rating = R[i,u]
                    qi =  Q.getrow(i)
                    pu = Ph.getcol(u)

                    prediction = qi.dot(pu)[0,0]
                    error = actual_rating - prediction

                    qi = qi + GAMMA_SGD * (error * pu.transpose()  -  LAMBDA * qi)
                    pu = pu + GAMMA_SGD * (error * qi.transpose()  -  LAMBDA * pu)

                    Q[i,:] = qi
                    Ph[:,k] = pu

            def BGD(Q,Ph,R, non_zero_indexes):
                """"In one epoch, Changes all rows in Q and columns in Ph related to all known entries R[i,u]
                """
                for i,u in non_zero_indexes:
                    actual_rating = R[i,u]
                    qi =  Q.getrow(i)
                    pu = Ph.getcol(u)
                    
                    prediction = qi.dot(pu)[0,0]
                    error = actual_rating - prediction

                    qi = qi + GAMMA_SGD * (error * pu.transpose()  -  LAMBDA * qi)
                    pu = pu + GAMMA_SGD * (error * qi.transpose()  -  LAMBDA * pu)

                    Q[i,:] = qi
                    Ph[:,k] = pu

            # Get the non-zero indexes 
            non_zero_rows, non_zero_columns = train_short_fat_sparse_matrix.nonzero()
            train_non_zero_indexes = list(zip(non_zero_rows, non_zero_columns))

            non_zero_count = len(non_zero_rows)
            print("train Non-zero entries count: ", non_zero_count)

            #use a lil_matrix this allows for efficient indexing 
            train_short_fat_sparse_matrix = lil_matrix(train_short_fat_sparse_matrix).astype(np.float64)
            def run_BGD(Q, Ph, R):
                """ Runs the BGD refining the Q and Ph matrices for i in range(0,EPOCHS)
                """
                Q = Q.copy().astype(np.float64)
                Ph = Ph.copy().astype(np.float64)
                R = train_short_fat_sparse_matrix

                rmse = []
                for i in range(0,EPOCHS):

                    print("Running BGD epoch {}".format(i))
                    BGD(Q,Ph,R, train_non_zero_indexes)

                    rmse.append(calculate_RMSE(Q, Ph, R, train_non_zero_indexes, non_zero_count))

                print('TRAIN BGD RMSE SERIES  {}'.format(k), rmse)
                return [Q, Ph, rmse]

            def run_SGD(Q, Ph, train_short_fat_sparse_matrix):
                """ Runs the SGD refining the Q and Ph matrices for i in range(0,EPOCHS)
                """
                rmse = []
                Q = Q.copy().astype(np.float64)
                Ph = Ph.copy().astype(np.float64)
                R = train_short_fat_sparse_matrix

                for i in range(0,EPOCHS):
                    '''
                    Running SGD with one item with nonzero entry at a time 
                    '''
                    print("Running SGD epoch {}".format(i))
                    i, u = train_non_zero_indexes[i]
                    SGD(i, u,Q,Ph,R )

                    rmse.append(calculate_RMSE(Q, Ph, R, train_non_zero_indexes, non_zero_count ))

                print('TRAIN SGD RMSE SERIES {}'.format(k), rmse)
                return [Q, Ph, rmse]

            ''' ----------------------------------------------------------------------------------------------------------------------------------------------------
            TASK 4 part 2: Accuracy calculation
            '''

            non_zero_rows, non_zero_columns = test_short_fat_sparse_matrix.nonzero()

            test_non_zero_indexes = list(zip(non_zero_rows, non_zero_columns))

            test_non_zero_count = len(non_zero_rows)
            print("test Non-zero entries count: ", test_non_zero_count)

            #use a lil_matrix this allows for efficient indexing 
            test_short_fat_sparse_matrix = lil_matrix(test_short_fat_sparse_matrix)

            '''
            BGD training and testing
            '''
            Q_trained, Ph_trained, rsme  = run_BGD(Q, Ph, train_short_fat_sparse_matrix)
            with open('training BGD rsme data {}.json'.format(k), 'w') as f:       
                json.dump(rsme, f)

            bgd_test_rmse_acurracy = calculate_RMSE(Q_trained, Ph_trained, test_short_fat_sparse_matrix, test_non_zero_indexes, test_non_zero_count)
            print("bgd_test_rmse with k {} = ".format(k), bgd_test_rmse_acurracy)
            test_BGD_accuracy[k] = bgd_test_rmse_acurracy

            '''
            SGD training and testing
            '''
            Q_trained, Ph_trained, rsme = run_SGD(Q, Ph, train_short_fat_sparse_matrix)
            with open('training SGD rsme data {}.json'.format(k), 'w') as f:       
                json.dump(rsme, f)
            
            sgd_test_rmse_acurracy = calculate_RMSE(Q_trained, Ph_trained, test_short_fat_sparse_matrix, test_non_zero_indexes, test_non_zero_count)
            print("sgd_test_rmse with k {}  =".format(k), sgd_test_rmse_acurracy)
            test_SGD_accuracy[k] = sgd_test_rmse_acurracy



        with open('Test SGD rsme data.json', 'w') as f:       
                json.dump(test_SGD_accuracy, f)

        with open('Test BGD rsme data.json', 'w') as f:       
                json.dump(test_BGD_accuracy, f)  