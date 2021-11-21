

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import tweepy

#Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "PaTejXKfRCGvZknZgzM4p3gEb"
consumer_secret = "IRqa3pZEtOPejZoXBIWNTtzrNchP1N6qHFEQLAdTw5FKFjBrcR"
# access_token = "AAAAAAAAAAAAAAAAAAAAAJ6QVgEAAAAA3wJ5NuhQEkBQcRnf8trQ
# gMP5KIc%3D9NFmkwg2lIc1kJBjxepdkDv0EPWeuX0qnXRwB19fbAGTrPpfpC"
access_token = "1271942406431096832-qEPNYpDTCpDPK6eZr8gKJMGO3VdbuK"
access_secret = "Qzzr5Oc3IqwvJnHRnuhEdBPYod8BTbxFVM8nuiKQFfgvS"
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Search Terms
target_terms = ("@BBC", "@CBS", "@CNN", "@FoxNews", "@NYTimes")

# Array to hold sentiment
sentiments = []

# Declare variables for sentiments
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

# Variable for holding the oldest tweet
oldest_tweet = ""
i = 0
counter = 1


# Loop through all target users
for target in target_terms:
    i+=1
    
    # Pull 100 tweets
    public_tweets = api.user_timeline(target, count=100)
    
    # Loop through the 100 tweets
    for tweet in public_tweets:
        text=tweet["text"]
        compound = analyzer.polarity_scores(text)["compound"]
        pos = analyzer.polarity_scores(text)["pos"]
        neu = analyzer.polarity_scores(text)["neu"]
        neg = analyzer.polarity_scores(text)["neg"]

        # Add each value to the appropriate array
        sentiments.append({"User": target,
                           "text":text,
                       "Date": tweet["created_at"],
                       "Compound": compound,
                       "Positive": pos,
                       "Negative": neu,
                       "Neutral": neg,
                       "Tweets Ago": counter})
    counter+=1

    # Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)

#Save as a CSV file

sentiments_pd.to_csv("/home/zy/Sentiments.csv")
df = pd.read_csv("/home/zy/Sentiments.csv")

df.head()

# Step 2: Data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

documents = df['text'].values.astype("U")

vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

k = 3
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

df['cluster'] = model.labels_

df.head()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold._t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state we define this random state to use this value in TSNE which is a randmized algo.
RS = 25111993

# Importing matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# %matplotlib inline

# Importing seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

z = pd.DataFrame(model.labels_.tolist())
digits_proj = TSNE(random_state=RS).fit_transform(features)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 3))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(18):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

print(list(range(0,3)))
sns.palplot(np.array(sns.color_palette("hls", 3)))
scatter(digits_proj, model.labels_)
plt.savefig('home/zy/digits_tsne-generated_18_cluster.png', dpi=120)
plt.show()
