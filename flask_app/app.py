from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	

	if request.method == 'POST':
		my_prediction = 1
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)