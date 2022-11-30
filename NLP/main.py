from flask import Flask, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from get_tweets import make_twitter_call
from topic_modeling import do_topic_modeling
from flask_cors import CORS

# Initialize Flask app.
print("Initializing Flash app.")
APP = Flask(__name__)
CORS(APP)
print("\t--- Done!\n")

# Load all pre-trained machine learning models.
print("1/4: Initializing Sentiment Pipeline")
sentiment_pipeline = pipeline("text-classification", model='cardiffnlp/twitter-roberta-base-sentiment-latest')
print("\t--- Done!")
print("2/4: Initializing Emotion Pipeline")
emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')
print("\t--- Done!")
print("3/4: Initializing Topic Modeling Summary Pipeline")
topic_modeling_summary_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
print("\t--- Done!")
print("4/4: Initializing Summarization Pipeline")
summarization_pipeline = pipeline("summarization")
print("\t--- Done!")


def get_sentiments_and_emotions(data):
    tweets = list(data["clean_text"])
    tweets_sentiment = sentiment_pipeline(tweets)
    data["sentiment_label"] = [s['label'] for s in tweets_sentiment]
    data["sentiment_score"] = [s['score'] for s in tweets_sentiment]

    tweets_emotion = emotion_pipeline(tweets)
    data["emotion_label"] = [s['label'] for s in tweets_emotion]
    data["emotion_score"] = [s['score'] for s in tweets_emotion]

    return data


def do_text_summarization(data):
    # Append the 'rating' Column to the Dataset
    data['rating'] = data[['retweet_count', 'reply_count', 'like_count', 'quote_count']].astype(float).sum(1)

    top_16 = data.nlargest(16, "rating")
    tweets = list(top_16["clean_text"])
    tweets_summary = summarization_pipeline(" ".join(tweets))

    return data, tweets_summary


@APP.route("/search/<tags>/")
def hello_world(tags: str):
    # Perform minimal cleaning of the requested tags
    tags = [tag.strip() for tag in tags.strip().split(",") if len(tag.strip()) > 0]
    tags = " OR ".join(tags)

    print("Step 1: Getting tweets...")
    data = make_twitter_call(tags)
    print("--- Step 1 Done!")

    print("Step 2: Doing sentiment analysis...")
    data = get_sentiments_and_emotions(data)
    print("--- Step 2 Done!")

    print("Step 3: Doing topic modeling...")
    data, topics = do_topic_modeling(data, topic_modeling_summary_model)
    print("--- Step 3 Done!")

    print("Step 4: Doing text summarization...")
    data, tweets_summary = do_text_summarization(data)
    print("--- Step 3 Done!")

    # convert dates to strings
    data['created_at'] = data['created_at'].astype(str)

    # Combine the DataFrame and the topic clusters into one JSON file.
    json_data = {
        "tweets": data.to_dict(orient="records"),
        "topics": topics,
        "summary": tweets_summary[0]["summary_text"]
    }

    return jsonify(json_data)


if __name__ == "__main__":
    APP.run()
