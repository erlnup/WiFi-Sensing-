import torch 
import torch.nn as nn

class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


    def forward(self, x):
        outputs, (h_n, c_n) = self.LSTM(x)

        return h_n[-1]


#input_size is the number of columns, which must matche the last dim in a tensor, which is number of columns 
#lstm = LSTMEncoder(3,3,3)
#test_input = torch.randn(2, 5, 3)




