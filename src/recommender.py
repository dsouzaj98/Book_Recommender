from eda import *

content_data = books_data[['original_title','authors','average_rating']]
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