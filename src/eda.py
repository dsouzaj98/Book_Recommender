import plotly.graph_objs as go
from plotly.offline import  init_notebook_mode, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import plotly_express as px
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from cleaning import *

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

#create merged data
cross_books_data['book_title']=cross_books_data['book_title'].apply(lambda x:clean_text(x))
merge_data=pd.merge(cross_books_data, cross_ratings_data, on='isbn')
merge_data=merge_data.sort_values('isbn', ascending=True)


stop_words=set(STOPWORDS)
author_string = " ".join(books['authors'])
# title_string = " ".join(books['original_title'])
cross_author_string = " ".join(merge_data['book_author'].astype(str))
cross_title_string = " ".join(merge_data['book_title'].astype(str))
cross_publisher_string = " ".join(merge_data['publisher'].astype(str))

def wordcloud(string):
    wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(string)
    fig=plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(wc)