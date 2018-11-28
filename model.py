from gensim import models

class WordEmbedding:
    '''
    WordEmbeddingClass
    '''
    def __init__(self, mode='word2vec'):
        if mode=='word2vec':
            EmbeddingFunc = models.Word2Vec
        elif mode=='fasttext':
            EmbeddingFunc = models.FastText
        elif mode=='doc2vec':
            EmbeddingFunc = models.Doc2Vec
        else:
            print('error!!! please')
            return

        self.EmbeddingFunc = EmbeddingFunc

    def train(self, corpus, size=200, **kwargs):
        model = self.EmbeddingFunc(size=size, **kwargs)
        model.build_vocab(corpus)
        model.train(sentences=corpus, total_examples=model.corpus_count, epochs=1)
        self.model = model

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.EmbeddingFunc.load(path)
