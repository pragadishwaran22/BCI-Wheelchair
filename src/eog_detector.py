import numpy as np

class SimpleBlinkDetector:
    def __init__(self, channel_idx, fs=250, window_ms=200, threshold=50.0):
        self.ch = channel_idx
        self.fs = fs
        self.win = int(fs * window_ms / 1000)
        self.threshold = threshold

    def detect(self, epoch_eog):
        # epoch_eog : (n_times,) single EOG channel
        rms = np.sqrt(np.convolve(epoch_eog**2, np.ones(self.win)/self.win, mode='same'))
        return (rms > self.threshold).any()
