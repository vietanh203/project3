# imports
from flask import Flask, render_template, request
import nltk
import numpy as py
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from lxml import html
from googlesearch import search
from bs4 import BeautifulSoup

import Utils

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/greeting")
def greeting():
    return str("Hello! I'm Botty. I can answer everything. If you want to exit, type Bye!")


@app.route("/getresponse")
# function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(Utils.getResponse(userText.lower()))


if __name__ == "__main__":
    app.run(debug=True)
