import os
import random

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import preprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DO_CURRICULUM = True
CURRICULUM = [1, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7]

class CaptionningModel(Module):
    def __init__(self, encoder, decoder) -> None:
        super(CaptionningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

def train_one_batch(data, model, optimizer, criterion):
    """Trains one batch of the baseline model"""

    model.train(True)
    X, captions = data  # captions is [batch size, caption_length]
    X = X.to(DEVICE)
    captions = captions.to(DEVICE)

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

    return total_loss

def train_one_batch_attention(data, model, optimizer, criterion, teacher_forcing_prob=1):
    """Trains one batch of the attention model. Can specify teacher_forcing_prob to use curriculum learning."""

    model.train(True)
    X, captions = data
    X = X.to(DEVICE)
    captions = captions.to(DEVICE)

    optimizer.zero_grad()
    annotations = model.encoder(X)
    total_loss = 0

    hidden_state, cell_state = None, None
    for i in range(1, model.decoder.caption_length):
        if i == 1 or random.uniform(0, 1) < teacher_forcing_prob:
            previous_word = captions[:, i-1]
        else:
            previous_word = torch.argmax(output, dim=-1)
        previous_word = captions[:, i-1]
        output, _, hidden_state, cell_state = model.decoder(annotations, previous_word, hidden_state, cell_state)
        loss = criterion(output, captions[:, i])
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    model.train(False)
    return total_loss / (model.decoder.caption_length - 1)

def train_one_epoch(dataloader, model, optimizer, criterion, epoch_num=0):
    """Trains one epoch of either model. epoch_num is used in curriculum learning."""
    losses = []
    
    for data in tqdm(dataloader, leave=False):
        # loss = train_one_batch(data, model, optimizer, criterion)
        loss = train_one_batch_attention(data, model, optimizer, criterion, teacher_forcing_prob=1 if not DO_CURRICULUM else CURRICULUM[epoch_num])
        losses.append(loss.item())
    return sum(losses) / len(losses)

def train(train_ds, model, num_epochs=10, batch_size=32, lr=0.01, epoch_perc=0.05):
    """Trains the model."""
    if DO_CURRICULUM and len(CURRICULUM) != num_epochs:
        raise ValueError()
    optimizer = Adam(model.parameters(), lr=lr)
    # weights = torch.ones(len(model.decoder.vocab), dtype=torch.float)
    # weights[model.decoder.vocab["<null>"]] = 0.01

    print("Train set size:", len(train_ds))

    ds_len = len(train_ds)
    idx = torch.multinomial(torch.ones(ds_len, dtype=torch.float) / ds_len, int(ds_len * epoch_perc))
    # idx = torch.ones(1, dtype=torch.long) * 50
    dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(idx))

    print("Training examples used:", len(idx))

    criterion = CrossEntropyLoss()

    # scheduler = ExponentialLR(optimizer, 0.8)
    im, l = train_ds[idx[0]]
    print(train_ds.indices[idx[0]][0])
    print(preprocessing.rebuild_sentence(l, model.decoder.vocab))
    train_loss = torch.zeros(num_epochs)
    for i in range(num_epochs):
        train_loss[i] = train_one_epoch(dataloader, model, optimizer, criterion, epoch_num=i)

        torch.save(model, os.path.join("checkpoints", "check_att{}.pt".format(i+1)))
        print("Epoch {} - Train loss = {:.2f}".format(i + 1, train_loss[i]))
        # print(preprocessing.rebuild_sentence(inference(model, im.unsqueeze(0))[0], model.decoder.vocab))
        print(preprocessing.rebuild_sentence(inference_attention(model, im.unsqueeze(0))[0], model.decoder.vocab))
        # scheduler.step()
    return train_loss

def inference(model, X):
    """Inference for the baseline model."""
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

def inference_attention(model, X, return_weights=False):
    """Inference for the attention model. Can optionally return the attention weights for vizualizations."""
    annotations = model.encoder(X)
    batch_size = X.shape[0]

    output = torch.zeros((batch_size, model.decoder.caption_length), dtype=torch.long)
    output[:, 0] = model.decoder.vocab["<start>"]
    all_attention_weights = torch.zeros((batch_size, model.decoder.caption_length, 49))

    hidden_state, cell_state = None, None
    for i in range(1, model.decoder.caption_length):
        output_prob, attention_weights, hidden_state, cell_state = model.decoder(annotations, output[:, i-1], hidden_state, cell_state)
        all_attention_weights[:, i, :] = attention_weights
        output[:, i] = torch.argmax(output_prob, dim=-1)
    if return_weights:
        return output, all_attention_weights
    return output
