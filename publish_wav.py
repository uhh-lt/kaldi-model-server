import redis
from time import sleep
import soundfile as sf
import numpy as np
import time

red = redis.StrictRedis()

print(sys.argv)
signal, samplerate = sf.read(sys.argv[1])

signal = (signal * (2**15)).astype(np.int16)
# signal = signal[:, 0]
print(samplerate)
print(signal.shape)

audio_data_channel = 'asr_audio'

chunks = [signal[x:x+4096] for x in range(0, len(signal), 4096)]

for chunk in chunks:
    t1 = time.time()
    red.publish(audio_data_channel, np.array(chunk.reshape(-1)).tobytes())
    print("publish")
    t2 = time.time()
    sleep((4096.0 / 16000.0) - (t2 - t1))
