import numpy as np
import scipy.signal as sig


def power_spectra(s : np.ndarray,
                  hop_length : int,
                  n_fft : int,
                  window_type : str = "hamming",) -> np.ndarray:
    """ 
    Computes the power spectra of a signal s on windowed segments of size window_size, whose start times are separated by step_size.

    Parameters
    ----------
    s : np.ndarray
        The signal to compute the power spectra of.
    hop_length : int
        The number of samples to move the window by.
    n_fft : int
        The number of points to use in the FFT.
    window_type : str
        The type of window to use.

    Returns
    -------
    spectra : np.ndarray
        The power spectra of the signal (of shape (num_windows, n_fft), where num_windows = (len(s) - len(window)) // step_size + 1))).
    """

    window = sig.get_window(window_type, n_fft)

    spectra = []
    num_windows = (len(s) - len(window)) // hop_length + 1
    
    for i in range(num_windows):
        start = i * hop_length
        segment = s[start:start + n_fft] * window
        spectrum = np.fft.fft(segment, n=n_fft)[:n_fft//2]
        power_spectrum = np.abs(spectrum) ** 2
        spectra.append(power_spectrum)
    
    return np.array(spectra)



def power_spectrum_3d(s : np.ndarray,
                      hop_length : int,
                      n_fft : int,
                      window_type : str = "hamming",
                      eps : float = 1e-10) -> np.ndarray:
    """ 
    Computes the cumualtive power spectrum of a signal s using power spectra computed on different windowed segments.

    Parameters
    ----------
    s : np.ndarray
        The signal to compute the power spectra of.
    hop_length : int
        The number of samples to move the window by.
    n_fft : int
        The number of points to use in the FFT.
    window_type : str
        The type of window to use.
    eps : float
        A small value to add to the power spectra to avoid taking the log of zero.

    Returns
    -------
    cum_power_spectrum : np.ndarray
        The cumualtive power spectrum of the signal (of shape (num_windows, n_fft), where num_windows = (len(s) - len(window)) // step_size + 1))
    """
    spectra = power_spectra(s = s, hop_length = hop_length, n_fft = n_fft, window_type = window_type)
    cum_power_spectrum = np.sum(spectra, axis = 0) - np.concatenate((np.zeros_like(spectra[0:1, :]), np.cumsum(spectra, axis = 0)[:-1]))
    return np.log(cum_power_spectrum + eps)



def pole_estimator(s : np.ndarray,
                   Fs : int,
                   hop_length : int,
                   n_fft : int,
                   window_type : str = "hamming",
                   **kwargs) -> np.ndarray:
    """ 
    Estimates the poles of the resonant filter used to model a signal s.

    Parameters
    ----------
    s : np.ndarray
        The signal to estimate the poles of.
    Fs : int
        The sampling frequency of the signal.
    hop_length : int
        The number of samples to move the window by.
    n_fft : int
        The number of points to use in the FFT.
    window_type : str
        The type of window to use.
    **kwargs
        Additional arguments to pass to scipy.signal.find_peaks.

    Returns
    -------
    poles : np.ndarray
        The poles of the signal.
    """

    cum_power_spectrum = power_spectrum_3d(s = s, hop_length = hop_length, n_fft = n_fft, window_type = window_type)

    # Find the peaks in the cumulative power spectrum and estimate the resonant frequencies
    peaks = sig.find_peaks(cum_power_spectrum[0], **kwargs)[0]
    f_peaks = Fs * peaks / n_fft

    # Estimate the damping of the resonant frequencies
    damps = (cum_power_spectrum[0, peaks] - cum_power_spectrum[1, peaks]) / 2 / hop_length

    # Compute the poles of the resonant filter
    poles = np.exp(- damps) * np.exp(1j * 2 * np.pi * f_peaks / Fs)

    return poles