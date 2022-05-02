# netflix_movie_recommender

# 1. Read in the datasets -  Read the data into a matrix 

 1:  - > movei id 
1488844,3,2005-09-06 - > user id, rating, date (remove date)
## dimensions
17K - movies & 480k - users 

 load only a sparse matrix that allows memeory for none-zero values 
 
 ## 3 lists 
 ### - movieids 
 ### - userIds
 ### - ratings 
 ### Note: ith user, provided ith rating for ith movie 
 
  ### Create two matrices 
1. MOVIES rows and USERS columns,  - contents of the cells being the ratings given by the users for the movie
2. USERS rows and MOVIES columns -


### Note: Use Scipy sparse matrix

# Task 2 DimSum 

1. For this task, consider a matrix A that has the users in rows (each row is a different user), and movies in columns (each column is a movie). This is a tall (many users) skinny (few movies) matrix, on which DIMSUM is designed to operate.
2. Implement the DIMSUM algorithm, that will map your skinny A to a matrix of cosine similarities B, of size |movies| × |movies|.
3. Compute an approximation of AT A from B. Slide 118 of Lecture 4 will help you.
4. Compute the exact AT A operation on the tall skinny sparse matrix. You can use any matrix library here (such as SciPy.sparse). Compare the resulting AT A matrix with the approximated one (for instance by computing the average MSE over all the entries). Write these results in your report, and analyse the impact of γ (DIMSUM parameter) on the performance of DIMSUM.

You have to implement the pseudocode given in the slides yourself. You can use the random Python library to generate random numbers, and NumPy and SciPy to access the sparse matrix elements, perform summations, and compute vector norms.

# Task 3 

Implement a stochastic descent algorithm to optimise P and Q of an SVD (A= UΣVt), Q=U, Pt = ΣVt
 
