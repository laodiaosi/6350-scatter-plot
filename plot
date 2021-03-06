import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
#
import numpy as np
#
import pandas as pd
# We'll hack a bit with the t-SNE code in sklearn.
# from sklearn.utils.extmath import _ravel
# Random state we define this random state to use this value in TSNE which is a randmized algo.
import seaborn as sns
from elasticsearch import Elasticsearch
from pyspark.shell import spark
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# Importing sklearn and TSNE.
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist


esConn = Elasticsearch()
def plot():
    np.random.seed(19860801)
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N)) ** 2
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()

def getHashTag(text):
    if "BLM" in text.lower():
        return "#BLM"


def getSentimentValue(text):
    sentimentAnalyser = SentimentIntensityAnalyzer()
    polarity = sentimentAnalyser.polarity_scores(text)
    if(polarity["compound"] > 0):
        return "positive"
    elif(polarity["compound"] < 0):
        return "negative"
    else:
        return "neutral"

def getSentiment(time, rdd):
    test = rdd.collect()
    for i in test:
        esConn.index(index="hash_tags_sentiment_analysis",
                     doc_type="tweet-sentiment-analysis", body=i)

        def scatter(x, colors):
            # We choose a color palette with seaborn.
            palette = np.array(sns.color_palette("hls", 3))

            # We create a scatter plot.
            f = plt.figure(figsize=(32, 32))
            ax = plt.subplot(aspect='equal')
            sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=120,
                            c=palette[colors.astype(np.int)])
            # ???

            # plt.xlim(-25, 25)
            # plt.ylim(-25, 25)
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

def getScatterPlot(time,rdd):
    if (not rdd.isEmpty()):
        global idx
        spark_df = spark.creatDataFrame(rdd,["content","sentiment","hashtag"])
        df = spark_df.toPandas()
        # Step 1: Load the data
        # import pandas as pd
        # df = pd.read_csv("Movies_Dataset.csv")

        df.head()

        # Step 2: Data preprocessing
        # from sklearn.feature_extraction.text import TfidfVectorizer
        # from sklearn.cluster import KMeans


        documents = df['content'].values.astype("U")

        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.fit_transform(documents)

        k = 3
        model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        model.fit(features)

        df['cluster'] = model.labels_

        df.head()

        # Commented out IPython magic to ensure Python compatibility.
        # import numpy as np
        # from sklearn.manifold import TSNE
        # import pandas as pd
        # import seaborn as sns
        # from matplotlib import pyplot as plt
        # import os
        # from numpy import linalg
        # from numpy.linalg import norm
        # from scipy.spatial.distance import squareform, pdist
        #
        # # Importing sklearn and TSNE.
        # import sklearn
        # from sklearn.manifold import TSNE
        # from sklearn.datasets import load_digits
        # from sklearn.preprocessing import scale
        #
        # # We'll hack a bit with the t-SNE code in sklearn.
        # from sklearn.metrics.pairwise import pairwise_distances
        # from sklearn.manifold._t_sne import (_joint_probabilities,
        #                                      _kl_divergence)
        # # from sklearn.utils.extmath import _ravel
        # Random state we define this random state to use this value in TSNE which is a randmized algo.
        RS = 25111993

        # Importing matplotlib for graphics.
        # import matplotlib.pyplot as plt
        # import matplotlib.patheffects as PathEffects
        # import matplotlib
        # %matplotlib inline

        # Importing seaborn to make nice plots.
        # import seaborn as sns
        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 2.5})

        z = pd.DataFrame(model.labels_.tolist())
        digits_proj = TSNE(random_state=RS).fit_transform(features)
        print(list(range(0, 3)))
        sns.palplot(np.array(sns.color_palette("hls", 3)))
        plt.scatter(digits_proj, model.labels_)
        # plt.savefig('digits_tsne-generated_18_cluster.png', dpi=120)
        plt.show()

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: sentimentAnalysis.py <broker_list> <topic>", file=sys.stderr)
    #     exit(-1)
    spark = SparkSession.builder.getOrCteate()
    sc = spark.sparkContext
    ssc = StreamingContext(sc, 30)

    # brokers, topic = sys.argv[1:]
    kvs = KafkaUtils.createDirectStream(
        ssc, topics=['twitter'], kafkaParams = {'metadata.broker.list': 'localhost:5061'})
    tweets = kvs.map(lambda x: str(x[1].encode("ascii", "ignore"))).map(
        lambda x: (x, getSentimentValue(x), getHashTag(x))).map(lambda x: {"hashTag": x[2], "sentiment": x[1],"content": x[0]})
    tweets.foreachRDD(getSentiment)
    tweets.foreachRDD(getScatterPlot)
    ssc.start()
    ssc.awaitTermination()
