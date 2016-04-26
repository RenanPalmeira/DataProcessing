# -*- coding: utf8 -*-

from __future__ import print_function

import os, sys

base = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base)

import unittest
from processor import TwitterProcessing

pos_tweets = [
    ('Eu te acho maravilhosa','positiva'), 
    ('Ele é impossível! Não pode existir alguém igual', 'positiva')
]

neg_tweets = [
    ('Esta garota é feia','negativa'), 
    ('Isto está impossível de ser encontrado','negativa')
]

class ProcessorTest(unittest.TestCase):

	def setUp(self):
		samples = TwitterProcessing.tokenize(pos_tweets + neg_tweets)
		self.processor = TwitterProcessing(samples)

	def test_classify_neg(self):
		sample = self.processor.classify('Você é muito feio')
		self.assertEqual(sample, 'negativa')

	def test_classify_pos(self):
		sample = self.processor.classify('eu te acho maravilhosa')
		self.assertEqual(sample, 'positiva')