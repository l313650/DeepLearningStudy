import torch
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.optim as optim

# Set up fields
TEXT = Field(tokenize=get_tokenizer("basic_english"), lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)

# Load the dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# Create iterators
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    sort_within_batch=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        flattened = self.flatten(embedded)
        output = self.fc(flattened)
        output = self.sigmoid(output)
        return output

vocab_size = len(TEXT.vocab)
embedding_dim = 32
max_len = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LogisticRegression(vocab_size, embedding_dim).to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

model.train()

for epoch in range(10):
    epoch_loss = 0
    epoch_acc = 0

    for batch in train_iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_iterator):.4f}')

model.eval()
epoch_acc = 0

with torch.no_grad():
    for batch in test_iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == batch.label).float()
        accuracy = correct.sum() / len(correct)
        epoch_acc += accuracy.item()

print(f'Test Accuracy: {epoch_acc/len(test_iterator):.2f}')

