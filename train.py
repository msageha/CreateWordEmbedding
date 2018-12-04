# coding:utf-8
import argparse
import os

import MeCab
from gensim.models.doc2vec import TaggedDocument

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

def load(path='../data/corpus', domains=['LB', 'OB', 'OC', 'OL', 'OM', 'OP', 'OT', 'OV', 'OW', 'OY', 'PB', 'PM', 'PN']):
    sentences = []
    for domain in domains:
        print(domain)
        for file in os.listdir(f'{path}/{domain}'):
            with open(f'{path}/{domain}/{file}') as f:
                for line in f:
                    sentence = tokenizer(line)
                    sentences.append(sentence)
    return sentences

def load_document(path='../data/corpus'):
    dataset = []
    domains = ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']#['LB', 'OB', 'OC', 'OL', 'OM', 'OP', 'OT', 'OV', 'OW', 'OY', 'PB', 'PM', 'PN']
    for domain in domains:
        print(domain)
        for file in os.listdir(f'{path}/{domain}'):
            with open(f'{path}/{domain}/{file}') as f:
                for i, line in enumerate(f):
                    tags = [file+'_line{0}'.format(str(i).zfill(4))]
                    sentence = tokenizer(line)
                    sen = TaggedDocument(words=sentence, tags=tags)
                    dataset.append(sen)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--type', default="Word2Vec", choices=['Word2Vec', 'FastText', 'Doc2Vec'], help='please specify learning type. word2vec or fasttext')
    parser.add_argument('--size', type=int, default=200, help='embedding vector size')
    parser.add_argument('--window', type=int, default=10, help='please specify window size')
    parser.add_argument('--min_count', type=int, default=5, help='please specify min count size of words')
    parser.add_argument('--save_name', type=str, help='save file name', required=True)
    parser.add_argument('--load_name', type=str, help='load file name', default='')
    parser.add_argument('--epochs', type=int, default=1, help='trainingepoch num')
    args = parser.parse_args()

    model = WordEmbedding(args.type)
    if args.load_name != '':
        domains = ['OC', 'OW', 'OY', 'PB', 'PM', 'PN']
        for domain in domains:
            model.load(path=f'../data/embedding/{args.type}/{args.load_name}.bin')
            corpus = load(path='../data/corpus', domains=[domain])
            model.retrain(corpus)
            model.save(path=f'../data/embedding/{args.type}/{args.save_name}_{domain}.bin')
    else:
        if args.type == 'Doc2Vec':
            corpus = load_document(path='../data/corpus')
        else:
            corpus = load(path='../data/corpus')
        model.train(corpus, size=args.size, window=args.window, min_count=args.min_count, epochs=args.epochs)
        model.save(path=f'../data/embedding/{args.type}/{args.save_name}.bin')
