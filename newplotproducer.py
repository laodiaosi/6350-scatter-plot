from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 40)  


def main():
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
    counts = stream(ssc, pwords, nwords, 40)
    make_plot(counts)


def make_plot(counts):
    pos=[]
    neg=[]
    for each in counts:
        if each!=[]:
            pos.append(each[0][1])
            neg.append(each[1][1])
    maxRange = max(max(pos), max(neg))
    time=list(range(len(pos)))
    plt.plot(time,pos,'bo-',label='positive')
    plt.plot(time,neg,'go-',label='negative')
    plt.axis([0-0.5,len(pos)+0.5,0,maxRange+50])
    plt.legend(loc='upper left')
    plt.ylabel('Word count')
    plt.xlabel('Time step')
    print ("***PLOTTING***")
    plt.savefig('plot.png')
    plt.show()


def load_wordlist(filename):
   """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    #words = sc.textFile(filename)
    #wordlist = words.flatMap(lambda x:x.encode('utf-8').split('\n'))
    #return wordlist.collect()
    words = {}
    f = open(filename,'rU')
    text = f.read()
    text = text.split('\n')
    for line in text:
        words[line] = 1
    f.close()
    return words


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    words = tweets.flatMap(lambda x:x.split(' '))
    def feel(word):
      if word in pwords:
            return "positive"
        if word in nwords:
            return "negative"
        else:
            return "neutral"
    pairs = words.map(lambda x: (feel(x.lower()),1))
    count = pairs.reduceByKey(lambda x,y:x+y)
    count = count.filter(lambda x: (x[0]=="positive" or x[0]=="negative"))

    def updateFunction(newValues, runningCount):
        if runningCount is None:
            runningCount = 0
        return sum(newValues, runningCount)

    runningCounts = count.updateStateByKey(updateFunction)
    runningCounts.pprint()

