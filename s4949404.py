# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

#SWITCHES
CF_user_user_text=1
CF_item_item_text=1


# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or 
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.
    
    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix), 
                    1: claculate similarity scores between items \
                        (columns of the matrix)
    
    Returns:
        similarity_dict (dictionary): dictionary where the keys are users 
        (or items) and the values are lists of tuples containing the most 
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to authomatically compute cosine similarity using
    an apposite function from any package, the computation should follow the 
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for 
    faster computation.
    """
    similarity_dict= {}
    
    # TO DO: Handle the absence of ratings (missing values in the matrix)
    data=np.array(matrix)
    nan_mask=np.isnan(data) #in this way we can look at the mask to check if we can compute the similarity
    

    # TO DO: If axis is 1, what do you need to do to calculate the similarity 
    # between items (columns)
    if axis==1:
        data=data.T
        nan_mask=nan_mask.T

    # TO DO: loop through each couple of entities to calculate their cosine 
    # similarity and store these results
    print(f"The data is a matrix {data.shape}")
    print(f"Computing the similarity matrix:")
    
    sim_matrix=np.empty([data.shape[0], data.shape[0]])   #initialize a matrix of similarity between users

    for (i,j) in tqdm(np.ndindex(sim_matrix.shape), total=sim_matrix.size):       #computate similarity matrix          
        if i==j:                                               #nan value on the diagonal
            sim_matrix[i,j]=0
        elif i<j:
            A,B=[], []                               #initialize vector that collect values only when both user rate something
            A_big, B_big=[],[]                         #initialize vector that collect all the rating of the user (to normalize later)
            for l in range(data.shape[1]):          
                if not nan_mask[i,l] and not nan_mask[j,l]:   #check both user rated the same item
                    A.append(data[i,l])
                    B.append(data[j,l])
                if not nan_mask[i,l]:
                    A_big.append(data[i,l])
                if not nan_mask[j,l]:
                    B_big.append(data[j,l])
            
            #####   Now we compute cosine similarity
            cosine=0
            if A and B:
                A,B=np.array(A),np.array(B)
                coord=np.dot(A,B)
                norm_A=np.linalg.norm(A_big)
                norm_B=np.linalg.norm(B_big)
            
                if norm_A!=0 and norm_B!=0:
                    cosine=coord/(norm_A*norm_B)
            sim_matrix[i,j]=cosine
            sim_matrix[j,i]=cosine


    #print(f"First 5 rows and column of the similarity matrix:")
    #print(sim_matrix[:5,:5])


    # TO DO: sort the similarity scores for each entity and add the top k most 
    # similar entities to the similarity_dict

    similarity_dict={i:[] for i in range(len(sim_matrix))}

    for user, similarity in enumerate(sim_matrix):  
        similarity_mask=[]
        ntuple=[]
        
        similarity_mask = np.logical_not(np.isnan(similarity))
        orderer=np.argsort(similarity)[::-1]
        
        for idx in orderer:
            if similarity_mask[idx] and len(ntuple) < k:
                ntuple.append((int(idx),float(similarity[idx])))

        similarity_dict[user] = ntuple

    if axis==1:
        print("For user 0 we got:")
        print(similarity_dict[0])

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    similar_users=np.array([pair[0] for pair in user_similarity[user_id]])
    cosine_similarity=np.array([pair[1] for pair in user_similarity[user_id]])

    numpy_user_item_matrix=np.array(user_item_matrix)
    user_rating=numpy_user_item_matrix[similar_users, movie_id]

    if CF_user_user_text==True:
        print(f"CF User-User: User ID:{user_id}, Movie ID: {movie_id}")
        print(user_similarity[user_id])
        print(similar_users)
        print(cosine_similarity)
        print(f"The rating for the movie {movie_id} for the users mentioned are respectively:")
        print(user_rating)
    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    mask=np.logical_not(np.isnan(user_rating))
    numerator = cosine_similarity[mask]@user_rating[mask]
    denominator =np.sum(cosine_similarity[mask])


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    if CF_user_user_text==True:
        print(f"\n The user {user_id}, rated the movie {movie_id} with a                      {user_item_matrix[user_id][movie_id]}")
        print(f"The predicted output with CF user-user is:            {predicted_rating}\n")
    return predicted_rating



def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target item
    similar_items=np.array([pair[0] for pair in item_similarity[movie_id]])
    cosine_similarity=np.array([pair[1] for pair in item_similarity[movie_id]])

    numpy_user_item_matrix=np.array(user_item_matrix)
    item_rating=numpy_user_item_matrix[user_id, similar_items]

    if CF_item_item_text==True:
        print(f"CF item-item: User ID:{user_id}, Movie ID: {movie_id}")
        print(item_similarity[movie_id])
        print(similar_items)
        print(cosine_similarity)
        print(f"The rating for the user {user_id} gave to the movie mentioned are respectively:")
        print(item_rating)

    # TO DO: implement item-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    mask=np.logical_not(np.isnan(item_rating))
    numerator = cosine_similarity[mask]@item_rating[mask]
    denominator =np.sum(cosine_similarity[mask]) 


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    if CF_user_user_text==True:
        print(f"\n The user {user_id}, rated the movie {movie_id} with a                      {user_item_matrix[user_id][movie_id]}")
        print(f"The predicted output with CF user-user is:            {predicted_rating}\n")

    return predicted_rating

# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)
    err_user_matrix=np.zeros(user_matrix.shape)
    err_item_matrix=np.zeros(item_matrix.shape)
    err_utility_matrix=np.zeros(utility_matrix.shape)
    mask=np.logical_not(np.isnan(utility_matrix))
    confront_err_utility=np.zeros(err_utility_matrix.shape)

    for step in tqdm(range(n_steps)):
        err_utility_matrix[mask] = (utility_matrix - user_matrix @ item_matrix.T)[mask]
        err_user_matrix = err_utility_matrix @ item_matrix
        err_item_matrix = (user_matrix.T @ err_utility_matrix).T


        user_matrix+=learning_rate*(err_user_matrix - regularization*user_matrix)
        item_matrix+=learning_rate*(err_item_matrix - regularization*item_matrix)

        if step%500==0:
            print(f"At step {step}, the ''accuracy'' of the utility matrix is: \n {np.sum((err_utility_matrix)**2)/utility_matrix.size}")



    return user_matrix, item_matrix


M = np.array([
    [2., 2., np.nan, 1., 4., 1., np.nan, 4., 5., np.nan],
    [np.nan, 3., 5., 3., np.nan, 5., 5., 4., 4., 2.],
    [5., 1., np.nan, np.nan, np.nan, np.nan, np.nan, 3., 2., 3.],
    [5., np.nan, 2., 4., 5., 4., 3., 3., np.nan, 4.],
    [1., 3., 5., 5., 4., np.nan, 5., np.nan, 5., 4.],
    [np.nan, np.nan, 2., 1., 3., 3., 1., 4., 4., 3.],
    [3., 5., 5., np.nan, 2., 4., np.nan, np.nan, 4., 5.],
    [np.nan, 1., 3., 3., 2., 1., 1., 5., 1., np.nan],
    [3., 1., np.nan, 4., 4., np.nan, 3., 5., 3., np.nan],
    [5., np.nan, np.nan, 4., 2., np.nan, 2., 2., 3., 4.]
])


if __name__ == "__main__":
    path = "u.data"       
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    #print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    #print(item_similarity_matrix.get(10,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13  
    movie_id = 100 

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
