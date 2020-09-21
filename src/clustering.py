import numpy as np
import pandas as pd 
from cleaning import ratings, books
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Merge the two tables then pivot so we have Users X Books dataframe. 
ratings_title = pd.merge(ratings, books[['book_id', 'title']], on='book_id' )
user_book_ratings = pd.pivot_table(ratings_title, index='user_id', columns= 'title', values='rating')

# Drop books that have fewer than 300 ratings.
user_book_ratings = user_book_ratings.dropna(axis='columns', thresh=300)
# Drop users that have given fewer than 100 ratings of these most-rated books
user_book_ratings = user_book_ratings.dropna(thresh=100)

# replace NaN's with zeroes for Truncated SVD
user_book_ratings_without_nan = user_book_ratings.fillna(0)
tsvd = TruncatedSVD(n_components=200, random_state=42)
user_book_ratings_tsvd = tsvd.fit(user_book_ratings_without_nan).transform(user_book_ratings_without_nan)

#view result in dataframe applying original indices
indices = user_book_ratings.index
book_ratings_for_clustering = pd.DataFrame(data=user_book_ratings_tsvd).set_index(indices)
book_ratings_training, book_ratings_testing = train_test_split(book_ratings_for_clustering, test_size=0.20, random_state=42)

#find the per-book ratings of test set
indices = book_ratings_testing.index
test_set_ratings = user_book_ratings.loc[indices]

mean_ratings_for_random_10 = []

# for each user, pick 10 books at random that the reader has rated and get the reader's average score for those books
for index, row in test_set_ratings.iterrows():
    ratings_without_nas = row.dropna()
    random_10 = ratings_without_nas.sample(n=10)
    random_10_mean = random_10.mean()
    mean_ratings_for_random_10.append(random_10_mean)

# get the mean of the users' mean ratings for 10 random books each    
mean_benchmark_rating = sum(mean_ratings_for_random_10) / len(mean_ratings_for_random_10)

#Kmeans prediction
clusterer_KMeans = KMeans(n_clusters=7).fit(book_ratings_training)
preds_KMeans = clusterer_KMeans.predict(book_ratings_training)

#silhouette score
kmeans_score = silhouette_score(book_ratings_training, preds_KMeans)

#GMM score
clusterer_GMM = GaussianMixture(n_components=25).fit(book_ratings_training)
preds_GMM = clusterer_GMM.predict(book_ratings_training)
GMM_score = silhouette_score(book_ratings_training, preds_GMM)

#compile predictions
indices = book_ratings_training.index
preds = pd.DataFrame(data=preds_KMeans, columns=['cluster']).set_index(indices)


# get a list of the highest-rated books for each cluster
def get_cluster_favorites(cluster_number):
    # create a list of cluster members
    cluster_membership = preds.index[preds['cluster'] == cluster_number].tolist()
    # build a dataframe of that cluster's book ratings
    cluster_ratings = user_book_ratings.loc[cluster_membership]
    # drop books that have fewer than 10 ratings by cluster members
    cluster_ratings = cluster_ratings.dropna(axis='columns', thresh=10)
    # find the cluster's mean rating overal and for each book
    means = cluster_ratings.mean(axis=0)
    favorites = means.sort_values(ascending=False)
    return favorites

# for each cluster, determine the overall mean rating cluster members have given books
def get_cluster_mean(cluster_number):
    # create a list of cluster members
    cluster_membership = preds.index[preds['cluster'] == cluster_number].tolist()
    # create a version of the original ratings dataset that only includes cluster members
    cluster_ratings = ratings[ratings['user_id'].isin(cluster_membership)]
    # get the mean rating
    return cluster_ratings['rating'].mean()

#cluster 0
cluster0_books_storted = get_cluster_favorites(0)
cluster0_mean = get_cluster_mean(0)

#cluster 1
cluster1_books_storted = get_cluster_favorites(1)
cluster1_mean = get_cluster_mean(1)

#cluster 2
cluster2_books_storted = get_cluster_favorites(2)
cluster2_mean = get_cluster_mean(2)

#cluster 3
cluster3_books_storted = get_cluster_favorites(3)
cluster3_mean = get_cluster_mean(3)

#cluster 4
cluster4_books_storted = get_cluster_favorites(4)
cluster4_mean = get_cluster_mean(4)

#cluster 5
cluster5_books_storted = get_cluster_favorites(5)
cluster5_mean = get_cluster_mean(5)

#cluster 6
cluster6_books_storted = get_cluster_favorites(6)
cluster6_mean = get_cluster_mean(6)



#associate test user with cluster
test_set_preds = clusterer_KMeans.predict(book_ratings_testing)
test_set_indices = book_ratings_testing.index
test_set_clusters = pd.DataFrame(data=test_set_preds, columns=['cluster']).set_index(test_set_indices)


mean_ratings_for_cluster_favorites = []
# put each cluster's sorted book list in an array to reference
cluster_favorites = [cluster0_books_storted, cluster1_books_storted, cluster2_books_storted, cluster3_books_storted, cluster4_books_storted, cluster5_books_storted, cluster6_books_storted]

# for each user, find the 10 books the reader has rated that are the top-rated books of the cluster. 
# get the reader's average score for those books
for index, row in test_set_ratings.iterrows():
    user_cluster = test_set_clusters.loc[index, 'cluster']
    favorites = cluster_favorites[user_cluster].index
    user_ratings_of_favorites = []
    # proceed in order down the cluster's list of favorite books
    for book in favorites:
        # if the user has given the book a rating, save the rating to a list
        if np.isnan(row[book]) == False:
            user_ratings_of_favorites.append(row[book])
        # stop when there are 10 ratings for the user
        if len(user_ratings_of_favorites) >= 10:
            break
    # get the mean for the user's rating of the cluster's 10 favorite books
    mean_rating_for_favorites = sum(user_ratings_of_favorites) / len(user_ratings_of_favorites)
    mean_ratings_for_cluster_favorites.append(mean_rating_for_favorites)
    
mean_favorites_rating = sum(mean_ratings_for_cluster_favorites) / len(mean_ratings_for_cluster_favorites)


#recommender function
def recommend(cluster_assignments=test_set_clusters, ratings_matrix=user_book_ratings, user_id):
    user_cluster = cluster_assignments.loc[user_id, 'cluster']
    favorites = get_cluster_favorites(user_cluster).index
    for book in favorites:
        # check if the user's rating for the book is NaN. 
        #If so, recommend the book. Otherwise, the user has already read this book, so move on
        if np.isnan(ratings_matrix.loc[user_id, book]):
            return book
    return null