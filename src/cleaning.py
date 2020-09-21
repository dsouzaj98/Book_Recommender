import pandas as pd 

#goodbooks data
books_data=pd.read_csv("data/goodbooks-10k-data/books.csv")
tags_data=pd.read_csv('data/goodbooks-10k-data/book_tags.csv')
ratings_data=pd.read_csv('data/goodbooks-10k-data/ratings.csv')
book_tags=pd.read_csv('data/goodbooks-10k-data/tags.csv')

#book crossing data
u_cols=['user_id','location','age']
cross_users_data=pd.read_csv('data/Book_reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1', low_memory=False, skiprows=1)

b_cols=['isbn', 'book_title', 'book_author','year_of_publication','publisher','img_s','img_m','img_1']
cross_books_data=pd.read_csv('data/Book_reviews/BX-Books.csv', sep=';', names=b_cols, encoding='latin-1', low_memory=False, skiprows=1)

r_cols=['user_id', 'isbn', 'rating']
cross_ratings_data=pd.read_csv('data/Book_reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1', low_memory=False, skiprows=1)

#drop unnecessary columns
# books_data = books_data.drop(columns=['id', 'best_book_id', 'work_id', 'isbn', 'isbn13', 'title','work_ratings_count',
#                                    'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 
#                                     'image_url','small_image_url'])

# books_data = books_data.dropna()
# cross_books_data = cross_books_data.drop(columns=['img_s', 'img_m', 'img_1'])

#remove duplicates
# ratings_data = ratings_data.sort_values("user_id")
# ratings_data.drop_duplicates(subset =["user_id","book_id"], keep = False, inplace = True) 
# books_data.drop_duplicates(subset='original_title',keep=False,inplace=True)
# book_tags.drop_duplicates(subset='tag_id',keep=False,inplace=True)
# tags_data.drop_duplicates(subset=['tag_id','goodreads_book_id'],keep=False,inplace=True)
# cross_ratings_data.drop_duplicates(subset =["user_id","isbn"], keep = False, inplace = True) 
# cross_books_data.drop_duplicates(subset='book_title',keep=False,inplace=True)



                            