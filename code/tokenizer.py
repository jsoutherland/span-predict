import numpy as np
from glove import Glove
from torchtext.data import get_tokenizer


class Tokenizer:
    def __init__(self, glove_embeds: Glove):
        self.base_tokenizer = get_tokenizer("basic_english")
        self.pad_token = '[PAD]'
        self.pad_ind = 0
        self.unk_token = '[UNK]'
        self.unk_ind = 1

        self.n_special_tokens = 2

        n_glove_dims = glove_embeds.n_dims
        self.n_dims = n_glove_dims
        self.glove = glove_embeds
        self.pad_vec = np.zeros(n_glove_dims)
        self.unk_vec = glove_embeds.unk_vector

        self.word2idx = {self.pad_token: self.pad_ind, self.unk_token: self.unk_ind}
        self.idx2word = [0 for _ in range(self.n_special_tokens)]
        self.idx2word[self.pad_ind] = self.pad_token
        self.idx2word[self.unk_ind] = self.unk_token

        assert self.pad_token not in self.glove.word2idx, f'{self.pad_token} token is invalid, it is in glove'
        assert self.unk_token not in self.glove.word2idx, f'{self.unk_token} token is invalid, it is in glove'

        for glove_word in self.glove.idx2word:
            idx = len(self.idx2word)
            self.idx2word.append(glove_word)
            self.word2idx[glove_word] = idx

    def build_nn_embedding_matrix(self):
        n_glove_vecs = len(self.glove.vectors)
        num_vecs = n_glove_vecs + self.n_special_tokens
        M = np.zeros((num_vecs, self.n_dims))
        # TODO: note: assuming pad ind is < n_special_tokens and leaving them all 0's
        M[self.unk_ind] = self.unk_vec
        M[self.n_special_tokens:, :] = np.array(self.glove.vectors)
        return M

    def tokenize(self, text: str, padding: int = None, truncate: int = None):
        tokens_raw = self.base_tokenizer(text)
        tokens = []
        token_ids = []
        mask = []
        for ind, t in enumerate(tokens_raw):
            if truncate is not None and ind >= truncate:
                break
            if t in self.word2idx:
                tokens.append(t)
                token_ids.append(self.word2idx[t])
            else:
                tokens.append(self.unk_token)
                token_ids.append(self.unk_ind)
            mask.append(1)
        if padding is not None:
            while len(mask) < padding:
                mask.append(0)
                tokens.append(self.pad_token)
                token_ids.append(self.pad_ind)

        return tokens, token_ids, mask
