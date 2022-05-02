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
 https://pantelis.github.io/cs634/docs/common/lectures/recommenders/netflix/

Perform the following sub-tasks:
• Implement Stochastic Gradient Descent using information from the article linked above.
• Add a (global) variable that allows to configure how many epochs of the algorithm will be performed. One epoch consists of computing and applying the gradient for all of the ratings in the Netflix dataset.
• Also implement Batch Gradient Descent, that should be only a small modification of what you have written in Point 2 above. Compare the accuracy obtained by Batch and Stochastic gradient descent, de- pending on the number of epochs and the gradient step, and write your findings in the report. Make plots in your report that show how the training accuracy evolves epoch after epoch, both for Stochastic and Batch Gradient Descent. Task 4 details how to measure accuracies.
Hint: A good gradient step for Batch Gradient Descent is 0.1. For Stochastic Gradient Descent, the step must be much smaller, about 1e-5 (0.00001).
Hint: The URL given above goes into detail explaining the gradients you need for both stochastic and batch gradient descent, and gives you the gradients of P and Q. The lecture slides provide a higher-level overview.
