# This is a wrapper for the Glove embeddings, it maps idx->word, word->idx, and idx->vector
import numpy as np


class Glove:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vectors = []
        self.unk_vector = None
        self.n_dims = None

    def load_embeddings(self, fname, n_dims):
        self.word2idx = {}
        self.idx2word = []
        self.vectors = []
        self.n_dims = n_dims

        with open(fname, 'rb') as f:
            for ind, l in enumerate(f):
                line = l.decode().split()
                assert len(line) == (1+n_dims), f'bad line {ind} length {len(line)} != {1+n_dims}'
                word = line[0]
                vect = np.array(line[1:], dtype=np.float64)
                self.word2idx[word] = ind
                self.idx2word.append(word)
                self.vectors.append(vect)
        self.unk_vector = np.mean(np.array(self.vectors), axis=0)
