
import os

import torch
import torch.nn.functional as F
from torch.nn import Module, LSTMCell, Embedding, Linear, Dropout

import preprocessing

class BasicDecoder(Module):
    def __init__(self, embedding_size, hidden_size, vocabulary, caption_length) -> None:
        super(BasicDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = len(vocabulary)
        self.vocab = vocabulary
        self.caption_length = caption_length
        self.dropout = Dropout(0.25)

        self.embedding = Embedding(self.vocab_size, embedding_size)
        self.lstm_cell = LSTMCell(embedding_size, hidden_size)
        self.linear_out = Linear(hidden_size, self.vocab_size)

    def forward(self, features, hidden_state, cell_state):
        batch_size = features.shape[0]
        
        if hidden_state is None:
            hidden_state = torch.zeros((batch_size, self.hidden_size))
        if cell_state is None:
            cell_state = torch.zeros((batch_size, self.hidden_size))

        hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

        output = torch.log_softmax(self.linear_out(self.dropout(hidden_state)), dim=-1)
        return output, hidden_state, cell_state

class AttentionModel(Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(AttentionModel, self).__init__()
        self.lin_annotations = Linear(input_size, hidden_size)
        self.lin_hidden = Linear(hidden_size, hidden_size)
        self.out = Linear(hidden_size, 1)
        self.dropout = Dropout(0.25)

    def forward(self, encodings_and_hidden_state):
        encodings, hidden_state = encodings_and_hidden_state   # encodings is [batch size, num_ann_vectors, context_size]
                                                                # prev_emb is [batch_size, hidden_sie]

        hidden_state = hidden_state.unsqueeze(1)        # prev_emb is [batch_size, 1, hidden_size]
        encodings_lin = self.lin_annotations(encodings) # encodings is [batch_size, num_ann_vectors, hidden_size]
        hidden_state = self.lin_hidden(hidden_state)     # prev_emb is [batch_size, 1, hidden_size]
        attention_input = torch.tanh(encodings_lin + hidden_state) # attention_input is [batch_size, num_ann_vectors, hidden_size]
        weights = F.softmax(self.out(attention_input), dim=-1)  # weights is [batch_size, num_ann_vectors]
        return torch.sum(weights * encodings, dim=1)  # return context vecotr [batch_size, content_size]

class AttentionDecoder(Module):
    def __init__(self, context_size, embedding_size, hidden_size, vocabulary, caption_length) -> None:
        super(AttentionDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab = vocabulary
        self.vocab_size = len(vocabulary)
        self.caption_length = caption_length

        self.embedding_layer = Embedding(self.vocab_size, embedding_size, self.vocab["<null>"])
        self.lstm_cell = LSTMCell(embedding_size + context_size, hidden_size)
        self.attention = AttentionModel(context_size, hidden_size)

        self.linear_hidden = Linear(hidden_size, embedding_size)
        self.linear_context = Linear(context_size, embedding_size)
        self.linear_out = Linear(embedding_size, self.vocab_size)

        self.cell_init = Linear(context_size, hidden_size)
        self.hidden_init = Linear(context_size, hidden_size)

        self.start_ohe = F.one_hot(torch.tensor(self.vocab["<start>"]), self.vocab_size)

    def forward(self, X):
        # X is [batch_size, num_vec, channels]
        batch_size = X.shape[0]

        cell_state = torch.tanh(self.cell_init(torch.mean(X, dim=1)))
        hidden_state = torch.tanh(self.hidden_init(torch.mean(X, dim=1)))

        output = torch.zeros((batch_size, self.caption_length, self.vocab_size))
        output[:, 0] = self.start_ohe

        for i in range(1, self.caption_length):
            previous_output = torch.argmax(output[:, i-1, :], dim=-1)   # [batch_size]
            previous_embedded = self.embedding_layer(previous_output)   # [batch_size, embedded_size]

            context = self.attention((X, hidden_state))    # [batch_size, context_size]

            lstm_input = torch.cat((previous_embedded, context), dim=1)

            hidden_state, cell_state = self.lstm_cell(lstm_input, (hidden_state, cell_state))
            
            combination = previous_embedded + self.linear_hidden(hidden_state) + self.linear_context(context)
            output[:, i] = F.log_softmax(self.linear_out(combination), dim=-1)

        return output

if __name__ == "__main__":
    captions = preprocessing.load_captions(os.path.join("dataset", "annotations", "annotations", "captions_train2017.json"))
    pre_captions = preprocessing.preprocess_captions(captions)
    vocab = preprocessing.create_vocabulary(pre_captions)
    print(len(vocab))

    decoder = BasicDecoder(1024, 512, 512, vocab, 10)

    random_x = torch.rand((1, 1024))
    output = decoder(random_x)
    print(output.shape)
    words = torch.argmax(output, dim=-1)
    print(preprocessing.rebuild_sentence(words[0], vocab))
