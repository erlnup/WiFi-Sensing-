import torch 
import torch.nn as nn

class BiLSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMEncoder, self).__init__()
        self.BiLSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True, bidirectional=True)


    def forward(self, x):
        outputs, (h_n, c_n) = self.BiLSTM(x)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=-1)
        return h
