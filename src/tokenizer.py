import numpy as np

class WordTokenizer:

    def __init__(self, text):
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        first_list = text.lower().split()
        self.vocab = sorted(list(set(first_list)))
        self.token_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_token = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text):
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        first_list = text.lower().split()
        return np.array([self.token_to_idx[w] for w in first_list])
        
    def decode(self, token_ids):
        list_of_tokens = []
        for i in range(len(token_ids)):
            list_of_tokens.append(self.idx_to_token[token_ids[i]])
        return " ".join(list_of_tokens)
