from LSTM_encoder import LSTMEncoder
from BiLSTM_encoder import BiLSTMEncoder 
from main import prepare_data
import asyncio
from utils.track_cpu_consumption import monitor_cpu
import threading
import torch.nn as nn



async def encoder():
    matrix, input_size = await prepare_data()

    print(f"matrix shape: {matrix.shape}")

    print(f"******* Starting up Encoder block *********\n")

    print(f"original matrix shape (batch_size/rows, sequence_length/length of each row array/timesteps, input size / features per time step): {matrix.shape}\n")
    print(f"original matrix shape (batch_size, sequence_length, input_size): {matrix.shape}\n")

    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_cpu, args=(stop_event,))
    monitor_thread.start()


    num_hidden_layers = 16
    num_stacked_lstm_encoders = 3
    print(f"Input to LSTM Encoder: input_size = {input_size}, num_hidden_layers = {num_hidden_layers}, num_stacked_lstm_encoders: {num_stacked_lstm_encoders}\n")
    
    lstm = LSTMEncoder(input_size,num_hidden_layers,num_stacked_lstm_encoders)
    batch_size, seg_length, input_size = matrix.shape

    output_lstm = lstm(matrix)

    print(f"Shape lstm output (batch_size, hidden_layers): {output_lstm.shape}\n")

    hidden_size = output_lstm.shape[1]

    encoded_lstm_seq = output_lstm.unsqueeze(1)
    
    print(f"Input to BiLSTMEncoder: hidden_size(num hidden layers in lstm) = {hidden_size}, num_hidden_layers = {num_hidden_layers}, num_stacked_lstm_encoders: {num_stacked_lstm_encoders}\n")
    biLstm_encoder = BiLSTMEncoder(hidden_size,num_hidden_layers,num_stacked_lstm_encoders)

    output_bilstm = biLstm_encoder(encoded_lstm_seq)

    print(f"Shape bilstm output: {output_bilstm.shape}\n")

    stop_event.set()
    monitor_thread.join()

    hidden_size_bilstm = output_lstm.shape[1]

    d_model = hidden_size_bilstm
    n_head = 4

    print(f"d_model = {d_model}   n_head = {n_head}\n")
    

    assert d_model % n_head == 0, "d_model (embedding layer) is not divisible by n_head"
    
    print(f"Input to Transformer Encoder Layer: d_model = {d_model}, n_head = {n_head}\n")
    transformer_encoder_layer = nn.TransformerEncoderLayer(d_model,n_head,batch_first=True)

    num_layers = 3
    
    print(f"Input to Transformer Encoder: num_layers = {num_layers}\n")
    transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

    output_transformer = transformer_encoder(output_lstm)

    print(f"Outputshape Transformer Encoder (batch_size,d_model/dimensionality): {output_transformer.shape}\n")


asyncio.run(encoder())

