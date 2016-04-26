# -*- coding: utf8 -*-

from __future__ import print_function

import nltk
from pymongo import MongoClient # About MongoDB https://goo.gl/PvVRcd

try:
    # Default in localhost 
    client = MongoClient() 
    db = client.twitter
    # Dump file in http://pastebin.com/aJjeY64C
    # For import dump file https://goo.gl/ae0xVB
    stopwords = [item['word'] for item in db.stopword.find()]
except Exception, e:
    stopwords = nltk.corpus.stopwords.words('portuguese')

class TwitterProcessing(object):
    """
    Reference in http://goo.gl/wjHNtm
    """
    classifier = None
    _word_features = None
    _training_set = None

    def __init__(self, samples, *args, **kwargs):
        self._word_features = self.get_word_features(self.get_word_in_sample(samples))
        self._training_set = nltk.classify.apply_features(self.extract_features, samples)
        self.classifier = nltk.NaiveBayesClassifier.train(self._training_set)

    def show_features(self):
        self.classifier.show_most_informative_features(self._word_features)

    def classify(self, sample):
        if type(sample) is str:
            sample = sample.split()
        return self.classifier.classify(self.extract_features(sample))

    def get_word_in_sample(self, sample):
        all_words = []
        for (word, sentiment) in sample:
            all_words.extend(word)
        return(all_words)

    def get_word_features(self, wordList):
        wordList = nltk.FreqDist(wordList)
        word_features = wordList.keys()
        return(word_features)

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self._word_features:
            features['contains(%s)' % word] = (word in document_words)
        return(features)

    @classmethod
    def tokenize(cls, phrases):
        global stopwords
        samples = []

        for (phrase, sentiment) in phrases:
            words_filtered = []
            wordList = nltk.wordpunct_tokenize(unicode(phrase.decode('utf-8')))

            for word in wordList:
                if word not in stopwords:
                    words_filtered.append(word)

            samples.append((words_filtered, sentiment))

        return samples
