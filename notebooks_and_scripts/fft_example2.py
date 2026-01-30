import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # ⟵ Erforderlich in PyCharm!!  Qt5Agg  TkAgg
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.fft
import pandas as pd
import os
from neo import io
# %%


frequency = 3
duration = 2
sampling_rate = 2000
N = sampling_rate * duration
x = np.linspace(0, duration, N, endpoint=False)
# frequencies = x * frequency
y = np.sin((2 * np.pi) * x * frequency)

noise = np.random.normal(0, 1, size=len(x))
y = y + noise * 0.1

fig=plt.figure(4)
plt.clf()
plt.plot(x, y)
plt.show()

normalized_y = normalize(y[:,np.newaxis], axis=0).ravel()

fig=plt.figure(5)
plt.clf()
plt.plot(x, normalized_y)
plt.show()


rffty = scipy.fft.rfft(normalized_y)
rfftx = scipy.fft.rfftfreq(N, 1 / sampling_rate)

fig=plt.figure(6)
plt.clf()
plt.plot(rfftx, np.abs(rffty))
plt.show()