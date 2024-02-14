# Getting Started with the Sentiment Analyser API


## Getting Started
In order to start the API application, the app.py file needs to be run.

With the command:
`python api.py`

<br>

## Configuration

The "getHashtagSentiment" uses the Twitter API to retrieve tweets which requires Twiter API keys. If you need to update the API keys this can be obtained via the following link: https://developer.twitter.com/

The twitter keys are stored in the TwitterAPI.py file

<br>

## API Endpoints

This project is a Python Flask API. Which exposes 2 HTTP POST endpoints:

1) http://localhost:3000/getSentenceSentiment
2) http://localhost:3000/getHashtagSentiment

Sample request body for getSentenceSentiment:

`{"text": "I hate broccoli"}`

Sample response for getSentenceSentiment:


`{"text": "I hate broccoli", "sentiment": "Negative"}`

Sample request body for getHashtagSentiment:

`{"hashtag": "Bitcoin"}`

Sample response for getHashtagSentiment:

`{"classified_tweets": [{"tweet": "RT @Hayess5178: $VRA @verasitytech On #Kucoin \n\n#Verasity Looking good, Following my
chart from 4 days ago. #VRA has broken $0.0075 resista\u2026", "username": "Simon Hayes", "sentiment": "Positive"},
{"tweet": "@CryptoLandEx @bridge_oracle @Bitcoin @ethereum @eth_classic @LTCFoundation @Cardano @EOSnFoundation
@CardanoFeed\u2026 https://t.co/x1aIOEPZS0", "username": "Kabiru Abubakar", "sentiment": "Positive"}, {"tweet": "they
never harvest and leak the DMs, it's always just postin' bullshit or bitcoin", "username": "Tom Joad The Wet Sprocket",
"sentiment": "Negative"}, {"tweet": "RT @WhaleWire: JUST IN: $XRP is now the #1 most popular altcoin in South Korea,
with trading volumes surging well beyond other cryptocurren\u2026", "username": "gravel", "sentiment": "Positive"}]}`

