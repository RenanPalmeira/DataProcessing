# -*- coding: utf8 -*-

from __future__ import print_function

import nltk
import processor

pos_tweets = [
    ('Eu te acho maravilhosa','positiva'), 
    ('Ela é linda demais','positiva'), 
    ('Ele é impossível! Não pode existir alguém igual', 'positiva')
]

neg_tweets = [
    ('Esta garota é feia','negativa'), 
    ('O mundo é cheio de desgraça','negativa'), 
    ('Isto está impossível de ser encontrado','negativa')
]

samples = processor.TwitterProcessing.tokenize(pos_tweets + neg_tweets)

if __name__=='__main__':
	processor = processor.TwitterProcessing(samples)
	print(processor.classify('Você é muito feio'))