 #######################
#			#
#   WORK IN PROGRESS	#
#			#
 #######################



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# Constants
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    # Load the WikiSQL dataset for training and validation
    train_data = load_dataset('wikisql', split='train+validation')

    # Load the WikiSQL dataset for testing
    test_data = load_dataset('wikisql', split='test')

    # Preprocess the data
    train_data = train_data.map(lambda example: {'question': example['question'], 'query': example['sql']['human_readable']})
    test_data = test_data.map(lambda example: {'question': example['question'], 'query': example['sql']['human_readable']})

    # Split the train+validation data into train and validation sets
    train_data = train_data.train_test_split(test_size=0.2, seed=42)
    train_data, valid_data = train_data['train'], train_data['test']

    return train_data, valid_data, test_data


class SQLDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row['question']
        target_query = row['sql']['human_readable']

        # Tokenize the input text
        input_encoding = self.tokenizer(
            question,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()

        # Tokenize the target SQL query
        target_encoding = self.tokenizer(
            target_query,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_query = target_encoding['input_ids'].squeeze()
        # print(input_ids.shape,attention_mask.shape,target_query.shape)
        return input_ids, attention_mask, target_query



# BERT-based encoder


class BERTBasedEncoder(nn.Module):
    def __init__(self):
        super(BERTBasedEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use pooled output as the encoder representation
        # print(pooled_output.shape)
        return pooled_output


# Seq2Seq decoder
class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, device):
        super(Seq2SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        #self.max_len = max_len
        self.device = device

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, hidden_size)  # Modified to concatenate features directly
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_output, target_seq):
        batch_size = encoder_output.size(0)
        target_len = target_seq.size(1)

        # Encoder hidden and cell states (no repeat needed)
        #encoder_hidden = encoder_output.squeeze(1)  # Remove unnecessary dimension
        encoder_hidden = encoder_output
        # Create an initial decoder hidden state
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)

        # Create an initial decoder cell state
        decoder_cell = torch.zeros(1, batch_size, self.hidden_size, device=self.device)

        outputs = torch.zeros(batch_size, target_len, self.output_size, device=self.device)

        # Start token
        input = target_seq[:, 0:-1]  # Exclude start token

        for t in range(target_len - 1):
            # Embedding
            embedded = self.embedding(input)

            # LSTM step
            lstm_output, (decoder_hidden, decoder_cell) = self.lstm(embedded, (decoder_hidden, decoder_cell))

            # Permute lstm_output (optional)
            #lstm_output_permuted = lstm_output.permute(1, 0, 2)
            lstm_output_permuted = lstm_output

            encoder_hidden = encoder_hidden.repeat(1, 511, 1)
            #encoder_hidden = encoder_hidden.repeat(1, lstm_output_permuted.size(1), 1)
            print("Encoder hidden shape:", encoder_hidden.shape)
            print("LSTM output shape:", lstm_output_permuted.shape)

            # Attention
            attention_scores = self.attention(torch.cat((encoder_hidden, lstm_output_permuted), dim=2))  # Concatenate features
            attention_weights = torch.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * encoder_hidden, dim=1)

            # Add an extra dimension to context_vector
            context_vector = context_vector.unsqueeze(1).repeat(1, lstm_output_permuted.size(1), 1)

            concat_input = torch.cat((lstm_output_permuted, context_vector), dim=2)
            output = self.out(self.attention_combine(concat_input).squeeze(1))

            # Next input is current target token
            input = target_seq[:, t + 1].unsqueeze(1)
            outputs[:, t + 1] = output

        return outputs


# NL2SQL model
class NL2SQLModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NL2SQLModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, attention_mask, target_query):
        # Encode input
        encoder_output = self.encoder(input_ids, attention_mask)

        # Decode the output
        decoder_output = self.decoder(encoder_output, target_query)

        # Compute loss
        loss = nn.CrossEntropyLoss()(decoder_output[:, 1:].view(-1, decoder_output.size(-1)), target_query[:, 1:].contiguous().view(-1))

        return loss


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, (input_ids, attention_mask, target_query) in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, target_query = input_ids.to(device), attention_mask.to(device), target_query.to(device)
        loss = model(input_ids, attention_mask, target_query)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Main function


def main():
    # Load data
    train_data, valid_data, test_data = load_data()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Get max sequence length from the dataset
    max_len = 512

    # Create datasets and dataloaders
    train_dataset = SQLDataset(train_data, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get the size of the vocabulary
    vocab_size = tokenizer.vocab_size

    # print("Vocabulary size:", vocab_size)

    # Initialize model, optimizer, and criterion
    encoder = BERTBasedEncoder().to(DEVICE)
    decoder = Seq2SeqDecoder(hidden_size=768, output_size=100, vocab_size=30522, device=DEVICE).to(DEVICE)
    model = NL2SQLModel(encoder, decoder).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()