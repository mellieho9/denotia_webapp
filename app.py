#importing libraries
import os
from flask.helpers import url_for
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, url_for

#creating instance of the class
app = Flask(__name__, template_folder='templates')

#to tell flask what url shoud trigger the function index()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/result/')
def result():
    return render_template('result.html')
@app.route('/faq/')
def faq():
    return render_template('faq.html')
@app.route('/about/')
def about():
    return render_template('about.html')
if __name__ == "__main__":
    app.run(debug=True)
