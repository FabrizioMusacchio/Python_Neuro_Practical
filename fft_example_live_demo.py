import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # ⟵ Erforderlich in PyCharm!!  Qt5Agg  TkAgg
import matplotlib.pyplot as plt
import scipy.fft

import os
from neo import io

# %% LIVE DEMO
file_path = "Data/Igor_1/"
file_names = [file for file in os.listdir(file_path) if file.endswith('.ibw')]
file_names = sorted(file_names)
print(f"file list (sorted): {file_names}")

test_file = os.path.join(file_path, file_names[6])
test_igor_read = io.IgorIO(test_file).read_analogsignal()

# time domain plot:
fig = plt.figure(1, figsize=(10,4))
plt.clf()
plt.plot(test_igor_read.times, test_igor_read, label=test_file)
plt.xlabel("time [ms]")
plt.legend(loc="best",fontsize=8)
plt.tight_layout()
plt.show()
#plt.savefig(file_path + " overview.pdf")

# Fourier Transformation
Current_signal = np.array(test_igor_read).flatten()
signal_rfft = scipy.fft.rfft(Current_signal)
Fs = np.array(test_igor_read.sampling_rate) # Sampling rate
N = np.shape(Current_signal)[0]
frequencies_rel = scipy.fft.rfftfreq(N, 1/Fs)

fig = plt.figure(2, figsize=(4,4))
plt.clf()
plt.plot(frequencies_rel, np.abs(signal_rfft), label='rfft')
plt.legend(loc="best",fontsize=8)
plt.xlabel("frequency [Hz]")
plt.ylabel("spectrum [V/Hz]")
plt.yscale("log")
plt.tight_layout()
plt.show()


# Power Spectrum:
pws = np.abs(signal_rfft)**2 / np.max(np.abs(signal_rfft)**2)
fig = plt.figure(3, figsize=(4,4))
plt.clf()
plt.plot(frequencies_rel, pws, label='power spectrumg')
plt.legend(loc="best",fontsize=8)
plt.xlabel("frequency [Hz]")
plt.ylabel("spectrum [V^2/Hz]")
plt.yscale("log")
plt.tight_layout()
plt.show()


# Spectogram:
fig = plt.figure(8, figsize=(7,5))
plt.clf()
plt.specgram(Current_signal, Fs=Fs)
plt.xlabel("time [s]")
plt.ylabel("frequency [Hz]")
plt.title("Spectogram")
plt.tight_layout()
plt.ylim((0, 50))
plt.show()



# Welch's method of FFT:
f, Pxx_spec = scipy.signal.welch(Current_signal, Fs, 'flattop', 1024, scaling='spectrum')
plt.figure(9, figsize=(5,5))
plt.clf()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear power spectrum [V RMS]')
plt.title('Power spectrum (scipy.signal.welch)')
plt.show()


plt.figure(10, figsize=(5,5))
plt.clf()
plt.psd(Current_signal, Fs=Fs)
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear power spectrum [V RMS]')
plt.title('Power spectrum (plt.psd())')
plt.show()