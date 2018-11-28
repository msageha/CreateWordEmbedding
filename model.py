from gensim import models

class WordEmbedding:
    '''
    WordEmbeddingClass
    '''
    def __init__(self, mode='word2vec'):
        if mode=='Word2Vec':
            EmbeddingFunc = models.Word2Vec
        elif mode=='FastText':
            EmbeddingFunc = models.FastText
        elif mode=='Doc2Vec':
            EmbeddingFunc = models.Doc2Vec
        else:
            print('error!!! please')
            return

        self.EmbeddingFunc = EmbeddingFunc

    def train(self, corpus, size=200, **kwargs):
        print('- - - satart training - - -')
        model = self.EmbeddingFunc(size=size, **kwargs)
        model.build_vocab(corpus)
        model.train(sentences=corpus, total_examples=model.corpus_count, epochs=1)
        self.model = model

    def retrain(self, corpus, **kwargs):
        self.model.train(sentences=corpus, total_examples=len(corpus), epochs=1)

    def save(self, path):
        print('- - - satart saving - - -')
        self.model.save(path)

    def load(self, path):
        print('- - - start loading - - -')
        self.model = self.EmbeddingFunc.load(path)
