from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from cleaning import ratings_data, books_data
import pandas as pd

ratings=ratings_data.groupby(['book_id']).mean()
u_ratings=ratings_data.groupby(['user_id']).mean()
book_id=ratings.index

author_dict=dict(zip( books_data.book_id,books_data.authors))
title_dict=dict(zip(books_data.book_id,books_data.original_title ))
r_author=ratings_data.copy()
r_author=r_author.rename(index=author_dict)


# book_id=ratings_data['book_id']
# rating=ratings_data['rating']
# rating=rating.to_numpy()
# book_id=ratings['book_id'])
def km(data,authors=r_author.index,clusters=8) :
    normalizer=Normalizer()
    kmeans=KMeans(n_clusters=clusters, max_iter=1000)
    pipeline=make_pipeline(normalizer,kmeans)
    pipeline.fit(data)
    labels=pipeline.predict(data)
    # author_df=pd.DataFrame({'labels':labels, 'author':authors})
    # # title_df=
    # return (author_df.sort_values('labels'))

# df=km(ratings)
