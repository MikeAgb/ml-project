
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

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

def train_one_batch(data, model, optimizer, criterion, captions_per_image):
    X, captions = data  # captions is [batch size, captions_per_image, caption_length]
    optimizer.zero_grad()

    # Make predictions
    output_captions = model(X)  # that has shape [batch size, caption_length, vocab_size]
    if captions_per_image > 1:
        output_captions = output_captions.unsqueeze(1).repeat(1, captions_per_image, 1, 1)  # Duplicate the output to do loss on multiple captions at once

    # Compute loss
    loss = compute_loss(output_captions, captions, criterion)
    loss.backward()
    optimizer.step()

    return loss

def train_one_epoch(train_ds, model, optimizer, criterion, captions_per_image, batch_size):
    losses = []
    
    ds_len = len(train_ds)
    idx = torch.multinomial(torch.ones(ds_len, dtype=torch.float) / ds_len, ds_len // 100)
    dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(idx))
    for data in tqdm(dataloader, leave=False):
        loss = train_one_batch(data, model, optimizer, criterion, captions_per_image)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def compute_val_loss(dataloader, model, criterion, captions_per_image):
    losses = []
    for X, y in tqdm(dataloader, leave=False):
        y_pred = model(X)
        if captions_per_image > 1:
            y_pred = y_pred.unsqueeze(1).repeat(1, captions_per_image, 1, 1)
        loss = compute_loss(y_pred, y, criterion).item()
        losses.append(loss)
    return sum(losses) / len(losses)

def train(train_ds, val_ds, model, captions_per_image=5, num_epochs=10, batch_size=32, lr=0.01):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    for i in range(num_epochs):
        avg_train_loss = train_one_epoch(train_ds, model, optimizer, criterion, captions_per_image, batch_size)
        #avg_val_loss = compute_val_loss(val_loader, model, criterion, captions_per_image)
        print("Epoch {} - Train loss = {:.2f}".format(i + 1, avg_train_loss))
