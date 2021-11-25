
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

        self.word_embedding = Embedding(self.vocab_size, embedding_size)
        self.lstm_cell = LSTMCell(embedding_size, hidden_size)

        self.linear_out = Linear(hidden_size, self.vocab_size)

    def forward(self, encoded_image):

        batch_size = encoded_image.shape[0]
        output = torch.zeros((batch_size, self.caption_length, self.vocab_size))

        # Give the encoded image as input to the LSTM cell with hidden state and cell state initialized as 0
        hidden_state = torch.zeros((batch_size, self.hidden_size))
        cell_state = torch.zeros((batch_size, self.hidden_size))
        hidden_state, cell_state = self.lstm_cell(encoded_image, (hidden_state, cell_state))

        # Start the sequence with the <start> token
        output[:, 0] = F.one_hot(torch.tensor(self.vocab["<start>"]), self.vocab_size)

        # sequences_ended = torch.tensor([False] * batch_size, dtype=torch.bool)

        # Add words to the sequence one by one
        for i in range(1, self.caption_length):
            previous_output_word = torch.argmax(output[:, i - 1, :], dim=-1)
            previous_embedded = self.word_embedding(previous_output_word)
            hidden_state, cell_state = self.lstm_cell(previous_embedded, (hidden_state, cell_state))
            output[:, i, :] = F.log_softmax(self.linear_out(hidden_state), dim=-1)

            # If a sequence already has a <end> token, then overwrite output with <null>
            # output[sequences_ended, i, :] = F.one_hot(torch.tensor(self.vocab["<null>"]), self.vocab_size).float()
            # sequences_ended = torch.logical_and(sequences_ended, torch.argmax(output[:, i, :], dim=-1) == self.vocab["<end>"])

        # End each sequence with a <end> token if it did already end
        # output[sequences_ended, self.caption_length - 1, :] = F.one_hot(torch.tensor(self.vocab["<null>"]), self.vocab_size).float()
        # output[torch.logical_not(sequences_ended), self.caption_length - 1, :] = F.one_hot(torch.tensor(self.vocab["<end>"]), self.vocab_size).float()

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
