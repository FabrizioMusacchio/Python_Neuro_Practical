# Create some dummy data:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # ⟵ Erforderlich in PyCharm!!  Qt5Agg  TkAgg
import matplotlib.pyplot as plt
import scipy.fft
import pandas as pd
import os
from neo import io
from sklearn.preprocessing import normalize


# define frequencies, amplitudes, and sampling rate and time array:
f1 =  2  # Frequency 1 in Hz
f2 = 10  # Frequency 2 in Hz
A1 = 6   # Amplitude 1
A2 = 2   # Amplitude 2
Fs = 100 # Sampling rate
t  = np.arange(0,1,1/Fs)

# calculate prime signals:
A_sin = A1 * np.sin(2 * np.pi * f1 * t)
A_cos = A2 * np.cos(2 * np.pi * f2 * t)
A_signal = A_sin + A_cos

# add some noise:
np.random.seed(1)
A_Noise = 2
Noise = np.random.randn(len(t)) * A_Noise
A_signal_noisy = A_signal + Noise

# plots:
fig=plt.figure(3, figsize=(9,4))
plt.clf()
plt.plot(t, A_sin, label="sine", lw=5, alpha=0.7)
plt.plot(t, A_cos, label="cosine", lw=5, alpha=0.7)
plt.plot(t, A_signal, lw=5, c="mediumorchid",
         label="superposition", alpha=0.75)
plt.plot(t, A_signal_noisy, lw=5, c="lime",
         label="superposition + noise", alpha=0.5)
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.xticks([0, 0.25, 0.5, 0.75, 1],
           ["0", r"$\frac{\pi}{2}$", r"$\pi$",
            r"$\frac{3}{4}\pi$", r"$2\pi$"])
plt.tight_layout()
plt.show()

# %%


A_signal_fft = scipy.fft.fft(A_signal)
A_signal_noisy_fft = scipy.fft.fft(A_signal_noisy)
frequencies = scipy.fft.fftfreq(np.size(t), 1/Fs)

fig=plt.figure(2)
plt.clf()
plt.stem(frequencies, np.abs(A_signal_fft))

frequency_eval_max = 100
A_signal_rfft = scipy.fft.rfft(A_signal, n=frequency_eval_max)
n = np.shape(A_signal_rfft)[0] # np.size(t)
frequencies_rel = n*Fs/frequency_eval_max * np.linspace(0,1,int(n))

fig=plt.figure(3)
plt.clf()
plt.stem(frequencies_rel, np.abs(A_signal_rfft))

def find_closest_within_array(array, value):
    arrays = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

filter_frequency = 10
# val, idx = find_closest_within_array(frequencies_rel, filter_frequency)
# A_signal_rfft[idx] = 0

# ll = 200 # lower amplitude limit
# ul = 500 # upper frequency limit
# A_signal_rfft[np.abs(A_signal_rfft)>ll][np.abs(A_signal_rfft[np.abs(A_signal_rfft)>ll])<ul] = 0
A_pass_limit = 50
A_signal_rfft[np.abs(A_signal_rfft)<A_pass_limit]=0
# A_signal_rfft[np.abs(A_signal_rfft)>ll]=0

fig=plt.figure(4)
plt.clf()
plt.stem(frequencies_rel, np.abs(A_signal_rfft))
plt.show()

A_signal_filtered = scipy.fft.irfft(A_signal_rfft)


fig=plt.figure(5, figsize=(9,5))
plt.clf()
plt.plot(t, A_sin, label="sine", lw=5)
plt.plot(t, A_cos, label="cosine", lw=5)
plt.plot(t, A_signal, lw=5, c="hotpink", label="superposition")
plt.plot(t, A_signal_filtered, c='lime',
         label="superposition filtered for $f=$"+str(filter_frequency)+" Hz")
# plt.plot(t, A_signal_noisy, lw=2, c="lime",
#          label="superposition + nois")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.xticks([0, 0.25, 0.5, 0.75, 1], ["0", r"$\frac{\pi}{2}$",
                                     r"$\pi$", r"$\frac{3}{4}\pi$", r"$2\pi$"])
plt.tight_layout()
plt.show()

# %% WITH FUNCTIONS

def frequency_filter(signal, filter_frequency, frequency_eval_max, Fs):
    signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    #signal_rfft = scipy.fft.rfft(signal)[:,0]
    n = np.shape(signal_rfft)[0]
    frequencies_rel = n * Fs / frequency_eval_max * np.linspace(0, 1, int(n))
    val, idx = find_closest_within_array(frequencies_rel, filter_frequency)

    signal_rfft_filtered = signal_rfft.copy()
    signal_rfft_filtered[idx] = 0
    signal_filtered = scipy.fft.irfft(signal_rfft_filtered)
    return signal_filtered, signal_rfft_filtered, signal_rfft, frequencies_rel

def amplitude_filter(signal, A_pass_limit, frequency_eval_max, Fs):
    signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    n = np.shape(signal_rfft)[0]
    frequencies_rel = n * Fs / frequency_eval_max * np.linspace(0, 1, int(n))
    signal_rfft_filtered = signal_rfft.copy()
    signal_rfft_filtered[np.abs(signal_rfft_filtered)<A_pass_limit]=0
    signal_filtered = scipy.fft.irfft(signal_rfft_filtered)
    return signal_filtered, signal_rfft_filtered, signal_rfft, frequencies_rel

def plot_comparison(frequencies_rel, Current_signal_rfft, Current_signal,
                    Current_signal_filtered, Current_signal_rfft_filtered,
                    A_sin, A_cos, A_signal, fignum=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=fignum, clear=True, figsize=(9, 8))
    ax1.stem(frequencies_rel, np.abs(Current_signal_rfft))
    ax1.set_title(r"|$X(f)|$")
    ax1.set_xlabel(r"frequency $f$ [Hz]")
    ax1.set_ylabel("amplitude [a.u.]")

    ax2.plot(t, A_sin, label="sine", lw=5, alpha=0.7)
    ax2.plot(t, A_cos, label="cosine", lw=5, alpha=0.7)
    ax2.plot(t, Current_signal, lw=5, c="lime", label="superposition + noise", alpha=0.5)
    ax2.plot(t, A_signal, lw=5, c="mediumorchid", label="superposition", alpha=0.75)
    ax2.legend(loc='upper right', fontsize=8)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_title(r"$x(t)$")
    ax2.set_xlabel(r"time $t$ [s]")

    ax3.stem(frequencies_rel, np.abs(Current_signal_rfft_filtered))
    ax3.set_title(r"|$X(f)|$")
    ax3.set_xlabel(r"frequency $f$ [Hz]")
    ax3.set_ylabel("amplitude [a.u.]")

    ax4.plot(t, A_sin, label="sine", lw=5, alpha=0.7)
    ax4.plot(t, A_cos, label="cosine", lw=5, alpha=0.7)
    ax4.plot(t, Current_signal, lw=5, c="lime", label="superposition + noise", alpha=0.5)
    ax4.plot(t, A_signal, lw=5, c="mediumorchid", label="superposition", alpha=0.75)
    ax4.plot(t, Current_signal_filtered, c='k',
             label="$X^{-1}(A_{signal, filtered})$")
    ax4.legend(loc='upper right', fontsize=8)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    ax4.set_title(r"$x(t)$")
    ax4.set_xlabel(r"time $t$ [s]")

    plt.tight_layout()
    plt.show()

A_pass_limit = 50
frequency_eval_max = 100
Current_signal = A_signal
Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
    amplitude_filter(signal=Current_signal, A_pass_limit=A_pass_limit, frequency_eval_max=100, Fs=Fs)

plot_comparison(frequencies_rel, Current_signal_rfft, Current_signal,
                Current_signal_filtered, Current_signal_rfft_filtered,
                    A_sin, A_cos, A_signal, fignum=1)


filter_frequency = 10
frequency_eval_max = 100
Current_signal = A_signal
Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
    frequency_filter(signal=Current_signal, filter_frequency=filter_frequency,
                     frequency_eval_max=100, Fs=Fs)

plot_comparison(frequencies_rel, Current_signal_rfft, Current_signal,
                Current_signal_filtered, Current_signal_rfft_filtered,
                    A_sin, A_cos, A_signal, fignum=2)

# %% REAL-WORLD TIME SERIES
""""""
# define frequencies, amplitudes, and sampling rate and time array:
f1 =  2  # Frequency 1 in Hz
f2 = 10  # Frequency 2 in Hz
A1 = 6   # Amplitude 1
A2 = 2   # Amplitude 2
A3 = 4   # Amplitude 2
Fs = 100 # Sampling rate
t  = np.arange(0,1,1/Fs)
t3  = np.arange(0,0.055,1/Fs)
peak_start_idx = 40

# calculate prime signals:
A_sin = A1 * np.sin(2 * np.pi * f1 * t)
A_cos = A2 * np.cos(2 * np.pi * f2 * t)
A_tan_peak = A3 * np.tan(2 * np.pi * f2 * t3)
A_tan = np.zeros(t.shape[0])
A_tan[peak_start_idx:peak_start_idx+A_tan_peak.shape[0]] = A_tan_peak
A_signal = A_sin + A_cos + A_tan

# add some noise:
np.random.seed(1)
A_Noise = 2
Noise = np.random.randn(len(t)) * A_Noise
A_signal_noisy = A_signal + Noise


frequency_eval_max = 100
A_signal_rfft = scipy.fft.rfft(A_signal, n=frequency_eval_max)
n = np.shape(A_signal_rfft)[0] # np.size(t)
frequencies_rel = n*Fs/frequency_eval_max * np.linspace(0,1,int(n))



A_signal_backFT = scipy.fft.irfft(A_signal_rfft)


# plots:
fig=plt.figure(3, figsize=(9,4))
plt.clf()
# plt.plot(t, A_sin, label="sine", lw=5, alpha=0.7)
# plt.plot(t, A_cos, label="cosine", lw=5, alpha=0.7)
plt.plot(t3, A_tan_peak, label="tangens peak", lw=5, alpha=0.7)
plt.plot(t, A_tan, label="tangens peak + phase", lw=5, alpha=0.7)
plt.plot(t, A_signal, lw=5, c="mediumorchid",
         label="superposition", alpha=0.75)
plt.plot(t, A_signal_backFT, c='lime',
         label="back FFT")
# plt.plot(t, A_signal_noisy, lw=5, c="lime",
#          label="superposition + noise", alpha=0.5)
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.xticks([0, 0.25, 0.5, 0.75, 1],
           ["0", r"$\frac{\pi}{2}$", r"$\pi$",
            r"$\frac{3}{4}\pi$", r"$2\pi$"])
plt.tight_layout()
plt.show()


fig=plt.figure(4)
plt.clf()
plt.stem(frequencies_rel, np.abs(A_signal_rfft))



filter_frequency = 30
frequency_eval_max = 100
Current_signal = A_signal
Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
    frequency_filter(signal=Current_signal, filter_frequency=filter_frequency,
                     frequency_eval_max=100, Fs=Fs)
frequencies = np.arange(11,52,1)
for frequency in frequencies:
    filter_frequency = frequency
    Current_signal = Current_signal_filtered
    Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
        frequency_filter(signal=Current_signal, filter_frequency=filter_frequency,
                         frequency_eval_max=100, Fs=Fs)
frequencies = np.arange(3,9,1)
for frequency in frequencies:
    filter_frequency = frequency
    Current_signal = Current_signal_filtered
    Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
        frequency_filter(signal=Current_signal, filter_frequency=filter_frequency,
                         frequency_eval_max=100, Fs=Fs)

plot_comparison(frequencies_rel, Current_signal_rfft, Current_signal,
                Current_signal_filtered, Current_signal_rfft_filtered,
                    A_sin, A_cos, A_signal, fignum=2)


# %% READING IGOR FILES

def frequency_filter_igor(signal, filter_frequency, Fs):
    # signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    #signal_rfft = scipy.fft.rfft(signal)
    # signal_rfft = scipy.fft.rfft(signal)[:,0]
    # n = np.shape(signal_rfft)[0]
    # frequencies_rel = n * Fs / frequency_eval_max * np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max / Fs * np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max /  np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max / np.arange(int(n))
    # frequencies_rel = np.linspace(0, frequency_eval_max, int(n))
    # T = 1.0 / Fs
    # frequencies_rel = np.arange(0, 0.5*Fs, Fs*1.0/n)
    # frequency_eval_max = 10
    # signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    # signal_rfft = scipy.fft.rfft(signal)
    signal_rfft = scipy.fft.rfft(signal)
    # n = np.shape(signal_rfft)[0]
    N = np.shape(signal)[0]
    frequencies_rel = scipy.fft.rfftfreq(N, 1/Fs)
    # frequencies_rel = np.arange(0, Fs, Fs * 1.0 / n)
    # fig = plt.figure(4)
    # plt.clf()
    # # plt.plot(frequencies_rel, np.abs(signal_rfft)/np.abs(signal_rfft).max())
    # plt.plot(frequencies_rel, np.abs(signal_rfft) )
    # plt.tight_layout()
    # plt.xlim([-0, 3])
    # plt.ylim([0, 0.065])
    # plt.show()

    # signal_rfft = scipy.fft.rfft(signal-signal.mean())[:-1]
    # frequencies_rel = np.arange(0, Fs-Fs * 1.0 / n, Fs * 1.0 / n)
    # pw = np.abs(signal_rfft) ** 2
    # pw = pw/np.max(pw)

    # fig = plt.figure(4)
    # plt.clf()
    # plt.plot(frequencies_rel, pw)
    # plt.xlim([-0, 3])
    # plt.ylim([0, 0.065])
    # plt.show()




    #frequencies_rel = n * Fs / n * np.linspace(0, 1, int(n))
    val, idx = find_closest_within_array(frequencies_rel, filter_frequency)

    signal_rfft_filtered = signal_rfft.copy()
    signal_rfft_filtered[idx] = 0
    signal_filtered = scipy.fft.irfft(signal_rfft_filtered)
    return signal_filtered, signal_rfft_filtered, signal_rfft, frequencies_rel


def frequency_filter_igor_range(signal, filter_frequency_range, Fs):
    # signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    #signal_rfft = scipy.fft.rfft(signal)
    # signal_rfft = scipy.fft.rfft(signal)[:,0]
    # n = np.shape(signal_rfft)[0]
    # frequencies_rel = n * Fs / frequency_eval_max * np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max / Fs * np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max /  np.linspace(0, 1, int(n))
    # frequencies_rel = frequency_eval_max / np.arange(int(n))
    # frequencies_rel = np.linspace(0, frequency_eval_max, int(n))
    # T = 1.0 / Fs
    # frequencies_rel = np.arange(0, 0.5*Fs, Fs*1.0/n)
    # frequency_eval_max = 10
    # signal_rfft = scipy.fft.rfft(signal, n=frequency_eval_max)
    # signal_rfft = scipy.fft.rfft(signal)
    signal_rfft = scipy.fft.rfft(signal)
    # n = np.shape(signal_rfft)[0]
    N = np.shape(signal)[0]
    frequencies_rel = scipy.fft.rfftfreq(N, 1/Fs)

    L = int(N*(filter_frequency_range[0]*1/Fs))
    R = int(N*(filter_frequency_range[1]*1/Fs))

    signal_rfft_filtered = signal_rfft.copy()
    signal_rfft_filtered[L:R] = 0
    signal_filtered = scipy.fft.irfft(signal_rfft_filtered)
    return signal_filtered, signal_rfft_filtered, signal_rfft, frequencies_rel



def plot_comparison_igor(frequencies_rel, t, Current_signal_rfft, Current_signal,
                    Current_signal_filtered, Current_signal_rfft_filtered, fignum=1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, num=fignum, clear=True, figsize=(9, 8))
    #ax1.stem(frequencies_rel, np.abs(Current_signal_rfft))
    ax1.plot(frequencies_rel, np.abs(Current_signal_rfft))
    ax1.set_title(r"|$X(f)|$")
    ax1.set_xlabel(r"frequency $f$ [Hz]")
    ax1.set_ylabel("amplitude [a.u.]")

    ax2.plot(t, Current_signal, lw=1, c="lime", label="signal", alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8)
    # plt.xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_title(r"$x(t)$")
    ax2.set_xlabel(r"time $t$ [s]")

    #ax3.stem(frequencies_rel, np.abs(Current_signal_rfft_filtered))
    ax3.plot(frequencies_rel, np.abs(Current_signal_rfft_filtered))
    ax3.set_title(r"|$X(f)|$")
    ax3.set_xlabel(r"frequency $f$ [Hz]")
    ax3.set_ylabel("amplitude [a.u.]")

    # ax4.plot(t, Current_signal, lw=5, c="lime", label="superposition + noise", alpha=0.5)
    ax4.plot(t, Current_signal, lw=1, c="lime", label="signal", alpha=0.5)
    ax4.plot(t, Current_signal_filtered, c='k', label="signal filtered")
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_title(r"$x(t)$")
    ax4.set_xlabel(r"time $t$ [s]")

    plt.tight_layout()
    plt.show()

# %% LIVE DEMO
file_path = "Data/Igor_1/"
file_names = [file for file in os.listdir(file_path) if file.endswith('.ibw')]
file_names = sorted(file_names)
print(f"file list (sorted): {file_names}")

test_file = os.path.join(file_path, file_names[3])
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



# %%


filter_frequency = 0.03
frequency_eval_max = 1
Current_signal = np.array(test_igor_read).flatten()
normalized_signal = normalize(Current_signal[:, np.newaxis], axis=0).ravel()
Current_signal = normalized_signal
Fs = np.array(test_igor_read.sampling_rate)
Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
    frequency_filter_igor(signal=Current_signal, filter_frequency=filter_frequency,
                          Fs=Fs)
plot_comparison_igor(frequencies_rel, test_igor_read.times, Current_signal_rfft, Current_signal,
                Current_signal_filtered, Current_signal_rfft_filtered, fignum=2)



filter_frequency_range = [0.5,40]
Current_signal = np.array(test_igor_read).flatten()
normalized_signal = normalize(Current_signal[:, np.newaxis], axis=0).ravel()
Current_signal = normalized_signal
Fs = np.array(test_igor_read.sampling_rate)
Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
    frequency_filter_igor_range(signal=Current_signal,
                                filter_frequency_range=filter_frequency_range, Fs=Fs)
plot_comparison_igor(frequencies_rel, test_igor_read.times, Current_signal_rfft, Current_signal,
                Current_signal_filtered, Current_signal_rfft_filtered, fignum=5)


pws = np.abs(Current_signal_rfft)**2 / np.max(np.abs(Current_signal_rfft)**2)

fig = plt.figure(7, figsize=(4,4))
plt.clf()
plt.plot(frequencies_rel, pws, label="power spectrum")
plt.xlabel("frequency [Hz]")
plt.ylabel("power spectrum [V^2/Hz]")
# plt.xscale("log")
plt.yscale("log")
plt.legend(loc="best",fontsize=8)
plt.tight_layout()
plt.show()


fig = plt.figure(8, figsize=(7,5))
plt.clf()
plt.specgram(Current_signal, Fs=Fs)
plt.xlabel("time [s]")
plt.ylabel("frequency [Hz]")
plt.title("Spectogram")
plt.tight_layout()
plt.show()



# signal.welch
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



# frequencies = np.arange(0.00,0.6,0.0001)
# for frequency in frequencies:
#     filter_frequency = frequency
#     Current_signal = Current_signal_filtered
#     Current_signal_filtered, Current_signal_rfft_filtered, Current_signal_rfft, frequencies_rel = \
#         frequency_filter_igor(signal=Current_signal, filter_frequency=filter_frequency,
#                          Fs=Fs)

# plot_comparison_igor(frequencies_rel, test_igor_read.times, Current_signal_rfft_tmp, normalized_signal,
#                 Current_signal_filtered, Current_signal_rfft_filtered, fignum=2)
