from eda import *
import numpy as np

content_data = books[['original_title','authors','average_rating']]
content_data = content_data.astype(str)

content_data['content'] = content_data['original_title'] + ' ' + content_data['authors'] + ' ' + content_data['average_rating']

content_data = content_data.reset_index()
indices = pd.Series(content_data.index, index=content_data['original_title'])

tfidf=TfidfVectorizer(stop_words='english')

tfidf_matrix=tfidf.fit_transform(content_data['authors'])
cosine_sim_author=linear_kernel(tfidf_matrix,tfidf_matrix)

def get_recommendations_books(title, cosine_sim=cosine_sim_author):
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim_author[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return list(content_data['original_title'].iloc[book_indices])

def author_book_shows(book):
    for book in book:
        print(book)

merge_data=merge_data[:40000]
book_rating=pd.pivot_table(merge_data, index='user_id', values='rating', columns='book_title', fill_value=0)

book_corr=np.corrcoef(book_rating.T)
book_list=list(book_rating)
book_titles=[]
for i in range(len(book_list)):
    book_titles.append(book_list[i])

def get_reco_collab(books_list):
    similar_books=np.zeros(book_corr.shape[0])
    for book in books_list:
        book_index=book_titles.index(book)
        similar_books +=book_corr[book_index]
    book_prefs=[]
    for i in range(len(book_titles)):
        book_prefs.append((book_titles[i], similar_books[i]))
    return sorted(book_prefs, key=lambda x:x[1], reverse=True)


list_of_books = ['one hundred years of solitude',
                 'stardust',
                 'mogs christmas',
                 'dragonmede',
                 'twopence to cross the mersey',
                 'the candywine development']
def get_final_reco(list1):
    test=get_reco_collab(list1)
    sim_books=[]
    i=0
    for n in range(10):
        sim_book=test[i][0]
        i+=1
        if sim_book in list1:
            continue
        else:
            sim_books.append(sim_book)
    return sim_books
    
