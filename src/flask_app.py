import boto3
import pandas as pd
import numpy as np
import sys
from io import StringIO
import surprise
from surprise import accuracy
from sklearn.model_selection import train_test_split
from surprise import SVD, NMF, KNNBaseline
import pickle
from flask import Flask, request
from flask import render_template
import pickle
import pandas as pd
from io import BytesIO
from recommender import *

app=Flask(__name__)

#home page
@app.route('/')
def get_new_data():
    return '''
    <body>  
    	<style>
        .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        }
		h1 {text-align: center;}
		h2 {text-align: center;}
		p {text-align: center;}
        body {background-image: url('https://images.squarespace-cdn.com/content/v1/55cd5e6de4b0af9801dd7aa7/1587662791718-G9371DYEL9H02TPQFBNU/ke17ZwdGBToddI8pDm48kL4WrIntsHuCODFzGytxs8sUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcw31z2cKmL83lZVTgYf1Shcnt0pzT4b-h8WwoQ3rX-86z0Q_QpJgDA4jmv5AtYw-J/SoHoViews.jpg?format=1500w');
                background-size: cover;
                background-repeat: no-repeat}
		form {text-align: center;}
		img {text-align: center;}
	</style>
        <div>
        <h1 style="color:White; background-color:Navy;">Book Recommender</h1>
        <h2>Find a new set of books to read!</h2>
        <div id="banner" style="overflow: hidden; display: flex; justify-content:space-around;">
        <form action="/predict-new" method='POST'>
          Book Name:<br>
          <input type="text" placeholder="Search.." name="searchterm"> 
          <br><br>
          <input type="submit" value="Find a new set!">
        </form>
        </div>
        </div>
        <'''
@app.route('/predict-new', methods=['GET','POST' ])
def predict():
    searchterm=request.form['searchterm']
    return ",".join(get_recommendations_books(f"{searchterm}"))

# @app.route('/predict-set', methods=['GET','POST'])
# def predict2():
#     searchterm2=request.form['searchterm2']
#     return ','.join(get_reco_collab(f"{searchterm2}"))
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)