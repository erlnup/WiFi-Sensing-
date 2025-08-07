import psutil
import time
import threading




def monitor_cpu(stop_event):

    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")
