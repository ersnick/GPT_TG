import tiktoken

class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken

    def count_tokens(self, input, model):
        res = self.tokenizer.encoding_for_model(model).encode(input)
        return len(res)