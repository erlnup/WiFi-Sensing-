from LSTM_encoder import LSTMEncoder
from BiLSTM_encoder import BiLSTMEncoder 
from main import prepare_data
import asyncio
from utils.track_cpu_consumption import monitor_cpu
import threading
async def encoding():
    matrix, input_size = await prepare_data()

    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_cpu, args=(stop_event,))
    monitor_thread.start()



    lstm = LSTMEncoder(input_size,248,3)
    batch_size, seg_length, input_size = matrix.shape
    print(f"Creating LSTMEncoder, batch size:{batch_size}, seg_length: {seg_length}, input_size: {input_size}")

    encoded_lstm = lstm(matrix)

    print(f"shape lstm output: {encoded_lstm.shape}")

    hidden_size = encoded_lstm.shape[1]

    encoded_lstm_seq = encoded_lstm.unsqueeze(1)

    BiLstm = BiLSTMEncoder(hidden_size,3,3)

    encoded_BiLstm = BiLstm(encoded_lstm_seq)

    stop_event.set()
    monitor_thread.join()


    print(encoded_BiLstm)



asyncio.run(encoding())
