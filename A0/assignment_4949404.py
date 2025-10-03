# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random

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
    sim_matrix=np.empty([data.shape[0], data.shape[0]])   #initialize a matrix of similarity between users

    for i,row in enumerate(data): 
        for j in range(i):
            mask = ~nan_mask[i, :] & ~nan_mask[j, :]   # both A and B have non-NaN values
            mask_a=~nan_mask[i, :]                      #only one has non-NaN value at time
            mask_b=~nan_mask[j, :]

            A = data[i, mask]   #A,B for the dot product
            B = data[j, mask]   

            dot_prod=data[i,mask] @ data[j,mask].T
            A_norm=np.linalg.norm(data[i,mask_a])
            B_norm=np.linalg.norm(data[j,mask_b])

            if np.isnan(A_norm*B_norm):
                cosine=0
            else:
                cosine=dot_prod/(A_norm*B_norm)
            sim_matrix[i,j]=cosine
            sim_matrix[j,i]=cosine
            sim_matrix[i,i]=0


    # TO DO: sort the similarity scores for each entity and add the top k most 
    # similar entities to the similarity_dict
    similarity_dict={i+1:[] for i in range(len(sim_matrix))}

    for user, similarity in enumerate(sim_matrix):  
        similarity_mask=[]
        ntuple=[]
        
        similarity_mask = np.logical_not(np.isnan(similarity))
        orderer=np.argsort(similarity)[::-1]
        
        for idx in orderer:
            if similarity_mask[idx] and len(ntuple) < k:
                ntuple.append((int(idx+1),float(similarity[idx])))

        similarity_dict[user] = ntuple

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
    similar_users=np.array([pair[0]-1 for pair in user_similarity[user_id]])
    cosine_similarity=np.array([pair[1] for pair in user_similarity[user_id]])

    numpy_user_item_matrix=np.array(user_item_matrix)
    user_rating=numpy_user_item_matrix[similar_users, movie_id]


    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    mask=np.logical_not(np.isnan(user_rating))
    numerator = cosine_similarity[mask]@user_rating[mask]
    denominator =np.sum(cosine_similarity[mask])


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

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
    similar_items=np.array([pair[0]-1 for pair in item_similarity[movie_id]])
    cosine_similarity=np.array([pair[1] for pair in item_similarity[movie_id]])

    numpy_user_item_matrix=np.array(user_item_matrix)
    item_rating=numpy_user_item_matrix[user_id, similar_items]

    # TO DO: implement item-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    mask=np.logical_not(np.isnan(item_rating))
    numerator = cosine_similarity[mask]@item_rating[mask]
    denominator =np.sum(cosine_similarity[mask]) 


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

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

    for step in range(n_steps):
        err_utility_matrix[mask] = (utility_matrix - user_matrix @ item_matrix.T)[mask]
        err_user_matrix = err_utility_matrix @ item_matrix
        err_item_matrix = (user_matrix.T @ err_utility_matrix).T


        user_matrix+=learning_rate*(err_user_matrix - regularization*user_matrix)
        item_matrix+=learning_rate*(err_item_matrix - regularization*item_matrix)

    return user_matrix, item_matrix

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
    print(user_similarity_matrix.get(1,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(100,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 1  
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
