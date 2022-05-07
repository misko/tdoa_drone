from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter#
from scipy import signal
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if len(sys.argv)!=3:
	print("%s input.wav distance_between_mics" % sys.argv[0])
	sys.exit(1)

input_wav_fname=sys.argv[1]
distance=float(sys.argv[2])




low_cut=25
high_cut=300

samplerate, data = wavfile.read(input_wav_fname)
distance_samples=int(samplerate*distance/341.0)+1

powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(data[:,0], Fs=samplerate,NFFT=4096*3)
plt.ylim([0,2000])
plt.show()
#sys.exit(1)




data=data[10000:]
print("sample_rate: %d, channels: %d" % (samplerate,data.shape[1]))

length = data.shape[0] / samplerate
print("length: %0.2fs" % length)

time = np.linspace(0., length, data.shape[0])
#plt.plot(time, data[:, 0], label="Left channel")
#plt.plot(time, data[:, 1], label="Right channel")
#plt.legend()
#plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
#plt.show()

#confirm that FFT shows peak at 1khz
sigA=data[:,0]
sigB=data[:,1]
sigA_filtered=butter_bandpass_filter(sigA,low_cut,high_cut,fs=samplerate)
sigB_filtered=butter_bandpass_filter(sigB,low_cut,high_cut,fs=samplerate)

start_ind=5*samplerate
end_ind=start_ind+2**14
spA = fftshift(fft(sigA[start_ind:end_ind]))
spB = fftshift(fft(sigB[start_ind:end_ind]))
t = np.arange(sigA[start_ind:end_ind].shape[0])
freq = fftshift(fftfreq(t.shape[-1]))
plt.plot(freq*samplerate, spA.real, freq*samplerate, spA.imag)
plt.plot(freq*samplerate, spB.real, freq*samplerate, spB.imag)
plt.xlim([-2000,2000])
plt.show()


spA_filtered_w = fftshift(fft(sigA_filtered[start_ind:end_ind]))
spB_filtered_w = fftshift(fft(sigB_filtered[start_ind:end_ind]))
t = np.arange(sigA_filtered[start_ind:end_ind].shape[0])
freq = fftshift(fftfreq(t.shape[-1]))

plt.plot(freq*samplerate, spA_filtered_w.real, freq*samplerate, spA_filtered_w.imag)
plt.plot(freq*samplerate, spB_filtered_w.real, freq*samplerate, spB_filtered_w.imag)
plt.xlim([-2000,2000])
plt.show()

window_length=2**13
window_step=2**10
pad=distance_samples
ar=[]
ar_filtered=[]
for window_start in range(3,data.shape[0]//window_step-3):
	#sigA=data[window_start:window_start+window_length,0]
	sigA_w=data[window_start*window_step-pad:window_start*window_step+window_length+pad,0]
	sigB_w=data[window_start*window_step:window_start*window_step+window_length,1]
	sigA_w_float=sigA_w.astype(np.float64)/sigA_w.max()
	sigB_w_float=sigB_w.astype(np.float64)/sigB_w.max()

	sigA_filtered_w=butter_bandpass_filter(sigA_w,low_cut,high_cut,fs=samplerate)
	sigB_filtered_w=butter_bandpass_filter(sigB_w,low_cut,high_cut,fs=samplerate)

	#sigA_filtered_w=sigA_filtered[window_start*window_step-pad:window_start*window_step+window_length+pad]
	#sigB_filtered_w=sigB_filtered[window_start*window_step:window_start*window_step+window_length]
	sigA_filtered_w_float=sigA_filtered_w.astype(np.float64)/sigA_filtered_w.max()
	sigB_filtered_w_float=sigB_filtered_w.astype(np.float64)/sigB_filtered_w.max()

	corr = signal.correlate(sigA_w_float,sigB_w_float,mode='valid')
	corr_filt = signal.correlate(sigA_filtered_w_float,sigB_filtered_w_float,mode='valid')

	ar.append(np.argmax(corr)-pad)
	ar_filtered.append(np.argmax(corr_filt)-pad)
plt.figure()
plt.plot(ar)
plt.plot(ar_filtered)
plt.ylim([-distance_samples,distance_samples])
plt.show()
