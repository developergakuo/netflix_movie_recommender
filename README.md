# netflix_movie_recommender

# Task 1: Read in the datasets -  Read the data into a matrix 
https://ai.vub.ac.be/irdm-project-2022/

### data format
 /1:  - > movei_id: 
 /1488844,3,2005-09-06 - > user id, rating, date (remove date)
### dimensions 
17K - movies & 480k - users 

 load only a sparse matrix that allows memory for none-zero values (Note: Use Scipy sparse matrix)
 ### read in strategy 
 3 lists  movieids, userIds, ratings 
 ### Note: ith user, provided ith rating for ith movie 
 
  ### Convert the lists into two matrices
1. MOVIES rows and USERS columns - short, wide matrix
2. USERS rows and MOVIES columns - tall, thin matrix

# Task 2: DimSum 

### this uses the tall but slim matrix - (movies in columns, and users in rows) - good fit for DIMSUM

1. For this task, consider a matrix A that has the users in rows (each row is a different user), and movies in columns (each column is a movie). This is a tall (many users) skinny (few movies) matrix, on which DIMSUM is designed to operate.
2. Implement the DIMSUM algorithm, that will map your skinny A to a matrix of cosine similarities B, of size |movies| × |movies|.
3. Compute an approximation of AT A from B. Slide 118 of Lecture 4 will help you.
4. Compute the exact AT A operation on the tall skinny sparse matrix. You can use any matrix library here (such as SciPy.sparse). Compare the resulting AT A matrix with the approximated one (for instance by computing the average MSE over all the entries). Write these results in your report, and analyse the impact of γ (DIMSUM parameter) on the performance of DIMSUM.

You have to implement the pseudocode given in the slides yourself. You can use the random Python library to generate random numbers, and NumPy and SciPy to access the sparse matrix elements, perform summations, and compute vector norms.

# Task 3: Gradient Descent Computation

### This uses the wide but short matrix (movies in rows, and users in columns) - bad fit for DIMSUM

Implement a stochastic descent algorithm to optimise P and Q of an SVD (A= UΣVt), Q=U, Pt = ΣVt
 https://pantelis.github.io/cs634/docs/common/lectures/recommenders/netflix/

Perform the following sub-tasks:
- Implement Stochastic Gradient Descent using information from the article linked above.
- Add a (global) variable that allows to configure how many epochs of the algorithm will be performed. One epoch consists of computing and applying the gradient for all of the ratings in the Netflix dataset.
- Also implement Batch Gradient Descent, that should be only a small modification of what you have written in Point 2 above. Compare the accuracy obtained by Batch and Stochastic gradient descent, de- pending on the number of epochs and the gradient step, and write your findings in the report. Make plots in your report that show how the training accuracy evolves epoch after epoch, both for Stochastic and Batch Gradient Descent. Task 4 details how to measure accuracies.

 * Hint: A good gradient step for Batch Gradient Descent is 0.1. For Stochastic Gradient Descent, the step must be much smaller, about 1e-5 (0.00001).
* Hint: The URL given above goes into detail explaining the gradients you need for both stochastic and batch gradient descent, and gives you the gradients of P and Q. The lecture slides provide a higher-level overview.

# Task 4: Accuracy
With your optimized P and Q produced in Task 3, you can now consider a new matrix M = QP t that will contain predicted ratings for every movie and every user. That matrix M would also consume about 30 GB of memory, so don’t compute it! Instead, feel free to multiply lines of Q and columns of P t manually to produce individual entries of M, that you can compare to the actual values in the original sparse matrix A that you produced in Task 1. This allows to measure the accuracy on the training set.

1.  Compute the RMSE (root mean-squared error) between the set of training movie-user pairs and their corresponding predictions. This is the training error.
2.   Split the sparse matrix A into a separate training and testing set, with the training set used to produce P and Q, and the testing set used to compute the RMSE. This is the testing error.
3.   (bonus) Read probe.txt to split the Netflix dataset into training and testing sets.


# Submission
The deadline for the project is Sunday the 5th of June 2022, at 23:59 Brussels Time. The project is to be submitted on Canvas and must be a zip file that contains:
1. One file that implements the 4 tasks above. It can be Python, Java, Scala, C++, C#, ... . Jupyter Notebooks are not allowed, but you can easily save a Jupyter Notebook as a Python file and submit it.
2. A 4-pages report that presents your results, for the different tasks, both in text and figures. For the DIMSUM γ parameters and the SVD K parameters (the number of eigenvalues), we ask you to compare at least 10 different values of these parameters. Feel free to use the rest of the pages to tell us what you want us to know about your project, for instance points that were particularly difficult to do, or a description of code that you think needs an explanation. We will have read your 4-pages report before your presentation in June at a time slot during the exam period, so use these 4 pages to "pre-brief" us on your project.
