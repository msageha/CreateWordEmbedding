# coding:utf-8
import argparse
import os

import MeCab

from model import WordEmbedding

tagger = MeCab.Tagger('')
tagger.parse('')

def tokenizer(text):
    wakati = []
    node = tagger.parseToNode(text).next
    while node.next:
        wakati.append(node.surface)
        node = node.next
    return wakati

def load(path='../data'):
    sentences = []
    domains = ['LB', 'OB', 'OC', 'OL', 'OM', 'OP', 'OT', 'OV', 'OW', 'OY', 'PB', 'PM', 'PN']
    for domain in domains:
        print(domain)
        for file in os.listdir(f'{path}/{domain}'):
            with open(f'{path}/{domain}/{file}') as f:
                for line in f:
                    sentence = tokenizer(line)
                    sentences.append(sentence)
    return sentences

def load_document(path='../data'):
    documents = []
    domains = ['OC']
    for domain in domains:
        print(domain)
        for file in os.listdir(f'{path}/{domain}'):
            document = []
            with open(f'{path}/{domain}/{file}') as f:
                for line in f:
                    sentence = tokenizer(line)
                    document += sentence
            sent = TaggedDocument(document, [file])
            document.append(sent)
    return documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('type' type=str, default="word2vec", help='please specify learning type. word2vec or fasttext')
    parser.add_argument('size', type=int, default=200, help='embedding vector size')
    parser.add_argument('window', type=int, default=10, help='please specify window size')
    parser.add_argument('min_count', type=int, default=5, help='please specify min count size of words')
    parser.add_argument('save_name', type=str, help='save file name')
    args = parser.parse_args()


model = WordEmbedding(args.type)

corpus = load(path='../data')
model.train(corpus, size=args.size, window=args.windows, min_count=args.min_count)
model.save(path=f'./embedding/{args.type}/{args.save_name}.bin')

