import torch
import torch.nn as nn

class TextGenerationRNN(nn.Module):
    """
    A GRU-based RNN model for next-word prediction.
    It takes a sequence of n-grams as input and predicts the next word.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(TextGenerationRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # GRU layer
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                          dropout=(dropout if num_layers > 1 else 0), 
                          batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass of the model.
        x shape: (batch_size, sequence_length)
        hidden shape: (num_layers, batch_size, hidden_size)
        """
        # Embedding
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embed_size)
        
        # GRU
        # output shape: (batch_size, sequence_length, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        
        # We only want the output from the last time step
        # output shape: (batch_size, hidden_size)
        output = output[:, -1, :] 
        
        # Dropout
        output = self.dropout(output)
        
        # Fully connected layer
        output = self.fc(output)  # Shape: (batch_size, vocab_size)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        """Initializes hidden state with zeros."""
        # Shape: (num_layers, batch_size, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
