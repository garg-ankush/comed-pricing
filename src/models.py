import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    """LSTM neural network"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, X, h=None):
        X, _ = X['X'], X['y']
        
        if h is None:
            h = (torch.zeros(1, X.size(0), 32).to(X.device),
                 torch.zeros(1, X.size(0), 32).to(X.device))
        
        output, hidden_state = self.lstm(X, h)
        last_hidden_state = output[:, -1, :]
        output = self.linear(last_hidden_state)
        output = F.relu(output)  # Apply ReLU activation
        return output

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        return self(batch)
