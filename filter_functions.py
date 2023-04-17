from scipy.signal import freqz, butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff1, fs1, order=2):
    nyq = 0.5 * fs1
    low = cutoff1 / nyq
    b, a = butter(order, [low], btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff1, fs1, order=2):
    b, a = butter_lowpass(cutoff1, fs1, order=order)
    filtered = lfilter(b, a, data)
    return filtered

def Filter_KIN(Kin_data):
    # Filters for EMG Sample rate and desired cutoff frequencies (in Hz).
    fs1 = 1000.0
    cutoff1 = 20.0
    passes1 = 2.0
    cutoff1 = cutoff1/(2**(1/passes1)-1)**(1/4.0)
    filt_order = 2.0
    b_KIN0_f1 = butter_lowpass_filter(Kin_data, cutoff1, fs1, order = filt_order)
    b_KIN0_f1_inv = b_KIN0_f1[::-1]
    b_KIN0_f2_inv = butter_lowpass_filter(b_KIN0_f1_inv, cutoff1, fs1, order = filt_order)
    b_KIN0_f2 = b_KIN0_f2_inv[::-1]
    return b_KIN0_f2