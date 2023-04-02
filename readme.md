# Movie rating prediction

# **Collaborative Filtering**

Collaborative filtering techniques are used to make recommendations based on user preferences and behavior. Collaborative filtering algorithms can be divided into two categories: memory-based and model-based. Memory-based algorithms use the entire dataset to generate recommendations, while model-based algorithms use a subset of the data to create a model that can be used to make predictions. Some of the most popular collaborative filtering algorithms include user-based collaborative filtering, item-based collaborative filtering, matrix factorization, and deep learning.

User-based collaborative filtering is a technique that recommends items based on the preferences of similar users. Item-based collaborative filtering is a technique that recommends items based on their similarity to items that a user has already rated. Matrix factorization is a technique that decomposes a large matrix into two smaller matrices that can be used to make predictions. Deep learning is a technique that uses neural networks to learn complex patterns in data.

# **Matrix factorization algorithm for collaborative filtering**

Matrix Factorization is a technique that decomposes a large matrix into two smaller matrices, one representing the users and the other representing the items. By multiplying these two matrices, we can reconstruct the original matrix and fill in the missing values. This way, we can predict how a user would rate an item that they have not seen before.

# Dataset

I built a collaborative filtering model to predict ratings that MovieLens users give to movies. I have used a dataset with 100836 ratings, 610 users, and 9724 movies.

Data can be downloaded using this command line:

```python
wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

The steps to build and train the Matrix Factorization model are as follows:

# 1. Encoding rating data

**Why we need to encode the data?**

Collaborative filtering is a technique for building recommender systems that use the ratings or preferences of users to predict what they might like. One of the challenges of collaborative filtering is that the user and item ids are often not continuous integers, but rather strings or other types of identifiers. This makes it difficult to use them as indices for matrices or tensors. To solve this problem, we need to encode the user and item ids into continuous integers that range from 0 to n-1, where n is the number of unique values.

To do this, I will use a simple function from the [fast.ai](http://fast.ai/) library called proc_col. This function takes a pandas column as input and returns a dictionary that maps each unique value to an integer, an array that contains the encoded values, and the number of unique values. For example, if we have a column with values ['a', 'b', 'c', 'a', 'b'], proc_col will return ({'a': 0, 'b': 1, 'c': 2}, [0, 1, 2, 0, 1], 3).

The first step is to encode the rating data into a sparse matrix, where each row represents a user and each column represents a movie. The matrix elements are the ratings given by the users to the movies. If a user has not rated a movie, the element is zero.

```python
def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
 
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)
```

```python
def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      df: a csv file with columns userId, movieId, rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    users_enc = proc_col(df.userId)
    num_users = len(np.unique(users_enc[1]))
    movie_enc = proc_col(df.movieId)
    num_movies = len(np.unique(movie_enc[1]))
    df.userId = users_enc[1]
    df.movieId = movie_enc[1]
    return df, num_users, num_movies
```

# 2. Initializing parameters

We need to specify the number of latent factors (k) that we want to use to represent the users and the items. The latent factors are hidden features that capture the preferences and characteristics of the users and the items. For example, a latent factor could represent how much a user likes comedy movies or how funny a movie is.

We also need to initialize two matrices: U and V. U is a n_users x k matrix that represents the users' latent factors. V is a n_movies x k matrix that represents the items' latent factors.

Here is an example of how the prediction matrix would look like with `7 users and 5 movies`

![Screen Shot 2023-04-01 at 5.26.43 PM.png](Movie%20rating%20prediction%20baf35cd6b0634434b64785664a3a5398/Screen_Shot_2023-04-01_at_5.26.43_PM.png)

# 3. Calculating the cost function

The cost function measures how well the model fits the data. It is defined as the mean squared error between the predicted ratings and the actual ratings. The predicted ratings are obtained by multiplying U and V.T (the transpose of V). The actual ratings are the non-zero elements of the rating matrix. We can use numpy to calculate the cost function.

Computes `mean square error` where first compute prediction. Prediction for user i and movie j is
emb_user[i]*emb_movie[j]

# 4.Calculating the gradient

The gradient is the direction that points to the minimum of the cost function. It tells us how to update the parameters to reduce the cost function. The gradient is composed of two parts: one for U and one for V. We can use `numpy` to calculate them.

# 5.Using gradient descent with momentum to fit the Matrix Factorization model

**What is gradient descent?** 

Gradient descent is a technique to find the optimal values of some parameters that minimize a loss function. It works by iteratively updating the parameters in the opposite direction of the gradient (the slope) of the loss function with respect to the parameters.

**How it is used in movie rating prediction?**

In movie rating prediction, gradient descent can be used to learn the latent features of users and movies that explain their ratings. For example, if we have a matrix of ratings given by users to movies, we can factorize it into two matrices: one that represents the association between users and features, and another that represents the association between movies and features. The features can be anything that influences the ratings, such as genre, actor, director, etc. The rating prediction is then a matrix multiplication or dot product of the user/movie features plus some biases.

**Explaining its parameters alpha and beta.**

The parameters that we want to optimize are the user/movie features and biases. We can initialize them randomly and then use gradient descent to update them based on the difference between the true ratings and the predicted ratings. The update rule for each parameter is:

parameter = parameter - alpha * gradient

where alpha is the learning rate, which controls how big of a step we take in each iteration. A small alpha means slow convergence but more accuracy, while a large alpha means fast convergence but more oscillation. The gradient is the partial derivative of the loss function with respect to the parameter, which tells us how much the loss function changes when we change the parameter slightly.

Another parameter that we can use in gradient descent is beta, which is the regularization term. Regularization is a technique to prevent overfitting, which means that the model learns too well from the training data but fails to generalize to new data. Regularization adds a penalty to the loss function based on the magnitude of the parameters, which shrinks them towards zero and reduces their complexity. The update rule for each parameter with regularization is:

parameter = parameter - alpha * (gradient + beta * parameter)

where beta is the regularization coefficient, which controls how much we penalize large parameters. `**A small beta means less regularization but more overfitting, while a large beta means more regularization but more underfitting.**`

In gradient descent, alpha refers to the learning rate, which determines the step size taken by the algorithm towards the minimum of the loss function. If alpha is too small, convergence may be slow. If alpha is too large, the algorithm may overshoot the minimum and fail to converge. Beta is not a standard parameter in gradient descent, but it can refer to the momentum term in some variations of the algorithm. The momentum term helps accelerate convergence and can prevent the algorithm from getting stuck in local minima.

```python
def gradient_descent(df, emb_user, emb_movie, iterations=100, \
			learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number 
				of iterations.Prints training cost and validation cost 
				(if df_val is not None) every 50 iterations.
    
    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    v_user, v_movie = 0, 0
    for i in range(iterations):
        grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
        v_user = 0.9 * v_user + (1 - 0.9) * grad_user
        v_movie = 0.9 * v_movie + (1 - 0.9) * grad_movie
        emb_user = emb_user - learning_rate * v_user
        emb_movie = emb_movie - learning_rate * v_movie
        if not i % 50:
            print(i, cost(df, emb_user, emb_movie), \
									cost(df_val, emb_user, emb_movie))
    return emb_user, emb_movie
```

# Results:

![Screen Shot 2023-04-01 at 6.15.19 PM.png](Movie%20rating%20prediction%20baf35cd6b0634434b64785664a3a5398/Screen_Shot_2023-04-01_at_6.15.19_PM.png)