#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/03/2023 10:52
# @Author  : Chengjie
# @File    : test_top2vec.py
# @Software: PyCharm
import os
import tempfile
import unittest
from top2vec import Top2Vec
import tensorflow_hub as hub


class Test_Top2Vec(unittest.TestCase):
    def test_top2vec(self):
        # model = Top2Vec([
        #     "The quick brown fox jumps over the lazy dog.",
        #     "I am a sentence for which I would like to get its embedding"],
        #     embedding_model='universal-sentence-encoder')
        from top2vec import Top2Vec
        from sklearn.datasets import fetch_20newsgroups

        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        print(type(newsgroups.data))
        document = ['fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B',
                    'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', 'fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B']

        model = Top2Vec(documents=document, embedding_model='universal-sentence-encoder', min_count=2)

        emb = model.embed(['fyvnzFlZPc', 'A'])
        print(emb)

    def test_tf_hub(self):
        print(os.path.join(tempfile.gettempdir(), "tfhub_modules"))
        # /var/folders/11/c8h_qyxn1sqfgs12w2ybr_r80000gn/T/tfhub_modules

        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        document1 = [
            "The quick brown fox jumps over the lazy dog.",
            "I am a sentence for which I would like to get its embedding"]
        document2 = ['fyvnzFlZPc', 'VsU9aWr3Sc', 'A', 'B', '1234']
        embeddings = embed(document2)

        print(embeddings)
