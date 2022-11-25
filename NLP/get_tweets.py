import tweepy
import re
import pandas as pd

with open("./bearer_token.txt") as file:
    BEARER_TOKEN = file.read()

client = tweepy.Client(bearer_token=BEARER_TOKEN)


def preprocess_tweet(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    # Remove RT tag and "@" username mentions.
    sentence = re.sub("(RT @\w+: )|(@\w+)", " ", sen)

    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", sentence)

    # Remove multiple spaces
    sentence = re.sub(r"\s{2,}", " ", sentence)

    # The previous RegEx rule may leave extra space at the end of the sentence.
    # This removes those extra spaces.
    return sentence.strip()


def make_twitter_call(tags):
    tweets = client.search_recent_tweets(
        query=f'({tags}) lang:en has:hashtags -is:reply',
        tweet_fields=['created_at', 'possibly_sensitive', 'public_metrics', 'entities'],
        max_results=100,
    )

    raw_tweets = [
        [
            t.id,
            t.text,
            preprocess_tweet(t.text),
            t.created_at,
            t.possibly_sensitive,
            t.public_metrics['retweet_count'],
            t.public_metrics['reply_count'],
            t.public_metrics['like_count'],
            t.public_metrics['quote_count'],
            [tag['tag'] for tag in t.entities["hashtags"]] if "hashtags" in t.entities else []
        ]
        for t in tweets.data
    ]

    df = pd.DataFrame(raw_tweets, columns=[
        "id", "text", "clean_text", "created_at", "is_sensitive",
        "retweet_count", "reply_count", "like_count", "quote_count",
        "hashtags"
    ])

    return df
