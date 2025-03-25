import numpy as np

def synthesis(excitation : np.ndarray,
              poles : np.ndarray,
              N : int) -> np.ndarray:
    """"
    Synthesizes a signal from an excitation signal and a set of poles.
    
    Parameters
    ----------
    excitation : np.ndarray
        The excitation signal.
    poles : np.ndarray
        The poles of the filter H(z).
    N : int
        The number of samples to use to compute the FFT of h and s.

    Returns
    -------
    s : np.ndarray
        The synthesized signal.
    """

    # Impulse response of the filter H(z) = sum 1/(1 - poles[i] / z)
    h = np.real(np.sum(poles[:, None] ** np.arange(N)[None, :], axis = 0))

    # DFT of the excitation signal E
    E = np.fft.fft(excitation, n = N)

    # DFT of the synthesized signal S
    S = E * np.fft.fft(h, n = N)

    # Inverse DFT of S
    s = np.real(np.fft.ifft(S))

    return s