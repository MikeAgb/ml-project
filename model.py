
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

def compute_loss(output, labels, criterion):
    return criterion(output.view(-1, output.shape[-1]), labels.view(-1))

def train_one_batch(data, model, optimizer, criterion):
    model.train(True)
    X, captions = data  # captions is [batch size, caption_length]
    optimizer.zero_grad()

    # Make predictions
    output_captions = model(X)  # that has shape [batch size, caption_length, vocab_size]
    # if captions_per_image > 1:
    #     output_captions = output_captions.unsqueeze(1).repeat(1, captions_per_image, 1, 1)  # Duplicate the output to do loss on multiple captions at once

    # Compute loss
    loss = compute_loss(output_captions, captions, criterion)
    loss.backward()
    optimizer.step()
    model.train(False)

    return loss

def train_one_epoch(dataloader, model, optimizer, criterion):
    losses = []  
    
    for data in tqdm(dataloader, leave=False):
        loss = train_one_batch(data, model, optimizer, criterion)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def compute_val_loss(dataloader, model, criterion):
    losses = []
    for X, y in tqdm(dataloader, leave=False):
        y_pred = model(X)
        # if captions_per_image > 1:
        #     y_pred = y_pred.unsqueeze(1).repeat(1, captions_per_image, 1, 1)
        loss = compute_loss(y_pred, y, criterion).item()
        losses.append(loss)
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
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    im, l = train_ds[idx[0]]
    print(train_ds.indices[idx[0]][0])
    print(preprocessing.rebuild_sentence(l, model.decoder.vocab))
    for i in range(num_epochs):
        avg_train_loss = train_one_epoch(dataloader, model, optimizer, criterion)
        #avg_val_loss = compute_val_loss(val_loader, model, criterion)
        print("Epoch {} - Train loss = {:.2f}".format(i + 1, avg_train_loss))
        print(preprocessing.rebuild_sentence(torch.argmax(model(im.unsqueeze(0))[0],-1), model.decoder.vocab))
        # scheduler.step()

