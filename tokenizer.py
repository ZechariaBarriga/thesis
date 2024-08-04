import torch

class Tokenizer(object):
    def __init__(self, data):
        # Define the additional tokens to be included
        self.additional_tokens = ['[C@H]', '[C@@H]', '[nH]', '[O-]', '[C@]', '[N+]', '[C@@]', '[S+]'] #[n+]

        # Build the unique character set
        unique_char = list(set(''.join(data))) + self.additional_tokens + ['<eos>'] + ['<sos>']
        self.test_char = unique_char
        self.mapping = {'<pad>': 0}

        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i

        self.inv_mapping = {v: k for k, v in self.mapping.items()}
        self.start_token = self.mapping['<sos>']
        self.end_token = self.mapping['<eos>']
        self.vocab_size = len(self.mapping.keys())

    def encode_smile(self, mol, add_eos=True):
        i = 0
        out = []
        while i < len(mol):
            matched = False
            for token in self.additional_tokens:
                if mol[i:i+len(token)] == token:
                    out.append(self.mapping[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                out.append(self.mapping[mol[i]])
                i += 1
        if add_eos:
            out.append(self.end_token)
        return torch.LongTensor(out)

    # testing purposes
    def decode_tensor(self, tensor):
        return ''.join([self.inv_mapping[token.item()] for token in tensor if token.item() != self.end_token])

    def batch_tokenize(self, batch):
        out = map(lambda x: self.encode_smile(x), batch)
        return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)

