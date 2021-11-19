import json
from kafka import SimpleProducer, KafkaClient
import tweepy
from tweepy import Stream
import configparser

# Note: Some of the imports are external python libraries. They are installed on the current machine.
# If you are running multinode cluster, you have to make sure that these libraries
# and currect version of Python is installed on all the worker nodes.

class TweeterStreamListener(tweepy.StreamListener):
    """ A class to read the twiiter stream and push it to Kafka"""

    def __init__(self, api):
        self.api = api
        super(tweepy.StreamListener, self).__init__()
        client = KafkaClient("localhost:9092")
        self.producer = SimpleProducer(client, async = True,
                          batch_send_every_n = 1000,
                          batch_send_every_t = 10)

    def on_status(self, status):
        """ This method is called whenever new data arrives from live stream.
        We asynchronously push this data to kafka queue"""
        msg =  status.text.encode('utf-8')
        #print(msg)
        try:
            self.producer.send_messages('twitterstream', msg)
        except Exception as e:
            print(e)
            return False
        return True

    def on_error(self, status_code):
        print("Error received in kafka producer")
        return True # Don't kill the stream

    def on_timeout(self):
        return True # Don't kill the stream

if __name__ == '__main__':

    # Read the credententials from 'twitter.txt' file
    config = configparser.ConfigParser()
    config.read('twitter.txt')
    #consumer_key = config['DEFAULT']['PaTejXKfRCGvZknZgzM4p3gEb']
    #consumer_secret = config['DEFAULT']['IRqa3pZEtOPejZoXBIWNTtzrNchP1N6qHFEQLAdTw5FKFjBrcR']
        #access_key = config['DEFAULT']['1271942406431096832-qEPNYpDTCpDPK6eZr8gKJMGO3VdbuK']
    #access_secret = config['DEFAULT']['Qzzr5Oc3IqwvJnHRnuhEdBPYod8BTbxFVM8nuiKQFfgvS']
    consumer_key = "PaTejXKfRCGvZknZgzM4p3gEb"
    consumer_secret = "IRqa3pZEtOPejZoXBIWNTtzrNchP1N6qHFEQLAdTw5FKFjBrcR"
    # access_token = "AAAAAAAAAAAAAAAAAAAAAJ6QVgEAAAAA3wJ5NuhQEkBQcRnf8trQ
    # gMP5KIc%3D9NFmkwg2lIc1kJBjxepdkDv0EPWeuX0qnXRwB19fbAGTrPpfpC"
    access_key = "1271942406431096832-qEPNYpDTCpDPK6eZr8gKJMGO3VdbuK"
    access_secret = "Qzzr5Oc3IqwvJnHRnuhEdBPYod8BTbxFVM8nuiKQFfgvS"

    # Create Auth object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # Create stream and bind the listener to it
    stream = tweepy.Stream(auth, listener = TweeterStreamListener(api))
    #stream = TweeterStreamListener(
     #       "PaTejXKfRCGvZknZgzM4p3gEb", "IRqa3pZEtOPejZoXBIWNTtzrNchP1N6qHFEQLAdTw5FKFjBrcR",
      #      "1271942406431096832-qEPNYpDTCpDPK6eZr8gKJMGO3VdbuK", "Qzzr5Oc3IqwvJnHRnuhEdBPYod8BTbxFVM8nuiKQFfgvS"
     #       )
    #Custom Filter rules pull all traffic for those filters in real time.
    #stream.filter(track = ['love', 'hate'], languages = ['en'])
    stream.filter(locations=[-180,-90,180,90], languages = ['en'])
