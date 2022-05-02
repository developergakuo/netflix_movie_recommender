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
 ### ratings 
 ### ith user, provided ith rating for ith movie 
 
  ### Create two matrices 
1. MOVIES rows and USERS columns,  - contents of the cells being the ratings given by the users for the movie
2. USERS columns and MOVIES rows -


# Use Scipy sparse matrix 
 
