#!/usr/bin/env python
# encoding: utf-8
import json
from TwitterAPI import TwitterAPI
from flask import Flask, request
from flask_cors import CORS, cross_origin
from SentimentAnalyser import SentimentAnalyser


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
SENTIMENT_ANALYSER = SentimentAnalyser()


@app.route('/getSentenceSentiment', methods=["POST"])
@cross_origin()
def get_sentence_sentiment():
    data = request.json
    text = data['text']
    sentiment = SENTIMENT_ANALYSER.analyse_sentiment(text)
    return json.dumps({'text': text,
                       'sentiment': sentiment})


@app.route('/getHashtagSentiment', methods=["POST"])
@cross_origin()
def get_hashtag_sentiment():
    data = request.json
    hashtag = data['hashtag']
    twitter_API = TwitterAPI()
    tweets = twitter_API.get_tweets(hashtag)

    json_response = {"classified_tweets": []}

    for tweet in tweets:
        classified_tweet = {"tweet": tweet.text, "username": tweet.user.name}
        sentiment = SENTIMENT_ANALYSER.analyse_sentiment(tweet.text)
        classified_tweet['sentiment'] = sentiment
        json_response['classified_tweets'].append(classified_tweet)

    return json.dumps(json_response)


app.run()
