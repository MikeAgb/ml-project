import os

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import preprocessing

class CaptionningModel(Module):
    def __init__(self, encoder, decoder) -> None:
        super(CaptionningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pretrained_encoding):
        encoding = self.encoder(pretrained_encoding)
        return self.decoder(encoding)

def train_one_batch(data, model, optimizer, criterion):
    model.train(True)
    X, captions = data  # captions is [batch size, caption_length]

    optimizer.zero_grad()

    X = model.encoder(X)

    total_loss = 0
    # Make predictions

    _, hidden_state, cell_state = model.decoder(X, None, None)
    for i in range(1, model.decoder.caption_length):
        features_input = model.decoder.embedding(captions[:, i-1])  # Use label for teacher forcing
        output, hidden_state, cell_state = model.decoder(features_input, hidden_state, cell_state)
        loss = criterion(output, captions[:, i])
        total_loss += loss

    # Compute loss
    total_loss.backward()
    optimizer.step()
    model.train(False)

    return loss

def train_one_epoch(dataloader, model, optimizer, criterion):
    losses = []  
    
    for data in tqdm(dataloader, leave=False):
        loss = train_one_batch(data, model, optimizer, criterion)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def train(train_ds, val_ds, model, num_epochs=10, batch_size=32, lr=0.01, epoch_perc=0.05):
    optimizer = Adam(model.parameters(), lr=lr)
    # weights = torch.ones(len(model.decoder.vocab), dtype=torch.float)
    # weights[model.decoder.vocab["<null>"]] = 0.01

    print("Train set size:", len(train_ds))

    ds_len = len(train_ds)
    idx = torch.multinomial(torch.ones(ds_len, dtype=torch.float) / ds_len, int(ds_len * epoch_perc))
    # idx = torch.ones(1, dtype=torch.long) * 50
    dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(idx))

    print("Training examples used:", len(idx))

    # counts = torch.zeros(model.decoder.vocab_size)
    # for _, captions in dataloader:
    #     # captions is shape [batch_size, word_per_caption]
    #     for caption in captions:
    #         for word in caption:
    #             counts[word] += 1
    # weights = max(counts) / counts

    criterion = CrossEntropyLoss()

    # scheduler = ExponentialLR(optimizer, 0.8)
    # im, l = train_ds[idx[0]]
    # print(train_ds.indices[idx[0]][0])
    # print(preprocessing.rebuild_sentence(l, model.decoder.vocab))
    for i in range(num_epochs):
        avg_train_loss = train_one_epoch(dataloader, model, optimizer, criterion)
        torch.save(model, os.path.join("checkpoints", "check{}.pt".format(i+1)))
        print("Epoch {} - Train loss = {:.2f}".format(i + 1, avg_train_loss))
        # print(preprocessing.rebuild_sentence(torch.argmax(model(im.unsqueeze(0))[0],-1), model.decoder.vocab))
        # scheduler.step()

def inference(model, X):
    X = model.encoder(X)
    batch_size = X.shape[0]

    _, hidden_state, cell_state = model.decoder(X, None, None)

    output = torch.zeros((batch_size, model.decoder.caption_length), dtype=torch.long)
    output[:, 0] = model.decoder.vocab["<start>"]

    for i in range(1, model.decoder.caption_length):
        features_input = model.decoder.embedding(output[:, i-1])  # Use label for teacher forcing
        output_prob, hidden_state, cell_state = model.decoder(features_input, hidden_state, cell_state)
        output[:, i] = torch.argmax(output_prob, dim=-1)
    return output
