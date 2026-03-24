class WordTokenizer:
    def __init__(self, text):
        text = text.replace(".", " . ").replace(",", " , ")
        words = text.lower().split()
        self.vocab = sorted(set(words))
        self.token_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_token = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        text = text.replace(".", " . ").replace(",", " , ")
        words = text.lower().split()
        return [self.token_to_idx[w] for w in words]

    def decode(self, token_ids):
        return " ".join(self.idx_to_token[int(i)] for i in token_ids)
