from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from cleaning import ratings_data, books_data
import pandas as pd

ratings=ratings_data.groupby(['book_id']).mean()
book_id=ratings.index

author_dict=dict(zip( books_data.book_id,books_data.authors))
title_dict=dict(zip(books_data.book_id,books_data.original_title ))
r_author=

# book_id=ratings_data['book_id']
# rating=ratings_data['rating']
# rating=rating.to_numpy()
# book_id=ratings['book_id'])
def km(data,author_dict=author_dict,clusters=8) :
    normalizer=Normalizer()
    kmeans=KMeans(n_clusters=clusters, max_iter=1000)
    pipeline=make_pipeline(normalizer,kmeans)
    pipeline.fit(data)
    labels=pipeline.predict(data)
    # author_df=pd.DataFrame({'labels':labels, 'author':author_dict})
    # # title_df=
    # return (df.sort_values('labels'))

# df=km(ratings)
