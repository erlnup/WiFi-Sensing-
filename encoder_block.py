
import torch
import torch.nn as nn
from LSTM_encoder import LSTMEncoder
from BiLSTM_encoder import BiLSTMEncoder
import threading
from utils.track_cpu_consumption import monitor_cpu

class Encoder(nn.Module):
    
    def __init__(self, input_size, num_hidden_layers=16, num_stacked_lstm_encoders=3, encoder_type="lstm"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_stacked_lstm_encoders = num_stacked_lstm_encoders
        self.output_dim = num_hidden_layers

        if self.encoder_type == "lstm":
            self.encoder = LSTMEncoder(input_size, num_hidden_layers, num_stacked_lstm_encoders)
        elif self.encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(input_size, num_hidden_layers, num_stacked_lstm_encoders)
        elif self.encoder_type == "transformer":
            d_model = input_size
            n_head = 4
            assert d_model % n_head == 0, "d_model must be divisible by n_head"
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, batch_first=True)
            self.encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, batch):
        # Optional: monitor CPU usage
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_cpu, args=(stop_event,))
        monitor_thread.start()

        output = self.encoder(batch)

        stop_event.set()
        monitor_thread.join()

        return output



