import numpy as np
from typing import List, Tuple


def frequency_domain_deconvolution(s : np.ndarray,
                                   poles : np.ndarray,
                                   N_fft : int,
                                   e_length : int) -> np.ndarray:
    """ 
    Estimates the excitation signal e which, when passed through a filter H(z), produces the signal s,
    where the filter H(z) is the sum-of-cosines filter defined by:
        `H(z) = sum_i 1/(1 - poles[i] / z)`

    Parameters
    ----------
    s : np.ndarray
        The signal to estimate the excitation signal of.
    poles : np.ndarray
        The poles of the filter H(z).
    N : int
        The number of samples to use to compute the FFT of h and s.
    e_length : int
        The length of the excitation signal.

    Returns
    -------
    e : np.ndarray 
        The estimated excitation signal.

    Notes
    -----
    N_fft should satisfy N_fft > len(s) + len(h) - 1. Thus, the impulse response of the filter H(z)
    (which should be infinite) is approximated by its first N_fft - len(s) samples.
    """

    # Impulse response of the filter H(z) = prod 1/(1 - poles[i] / z)
    h = np.real(np.sum(poles[:, None] ** np.arange(N_fft - len(s))[None, :], axis = 0))

    # DFT of s and h
    S = np.fft.fft(s, n = N_fft)
    H = np.fft.fft(h, n = N_fft)

    # DFT of the excitation signal E
    E = S / H

    # Inverse DFT of E
    e = np.real(np.fft.ifft(E))[:e_length]

    return e


def synchronize_and_scale(s : List[np.ndarray],
                filter_poles : List[np.ndarray],
                N_fft : int,
                e_length : int) -> Tuple[np.ndarray, List[float]]:
    e = []
    for i in range(len(s)):
        e.append(frequency_domain_deconvolution(s = s[i], poles = filter_poles[i], N_fft = N_fft, e_length = e_length))
    
    tau = np.zeros((len(e), len(e)))
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            r_ij = np.correlate(e[i], e[j], mode = "full")
            tau[i, j] = np.argmax(r_ij) - len(e[j]) + 1
            tau[j, i] = -tau[i, j]

    energies = []
    for i in range(len(e)):
        energies.append(np.sum(e[i] ** 2))
    
    return tau.astype(int), energies


def least_squares_deconvolution(s : List[np.ndarray],
                                filter_poles : List[np.ndarray],
                                N_fft : int,
                                e_length : int) -> np.ndarray:
    # For synchronization and scaling, we only need a short section at the strat of the excitation, so e_length = 1000 is fine
    tau, energies = synchronize_and_scale(s = s, filter_poles = filter_poles, N_fft = N_fft, e_length = 1000)    
    first = np.argmax(tau[0])

    H = []
    X = []
    for i in range(len(s)):
        h_i = np.real(np.sum(filter_poles[i][:, None] ** np.arange(N_fft)[None, :], axis = 0))

        # DFT of s and h
        t_i = - tau[first, i]
        delayed_s = np.concatenate((s[i][t_i:], np.zeros(t_i))) / np.sqrt(energies[i])
        S_i = np.fft.fft(delayed_s, n = N_fft)
        H_i = np.fft.fft(h_i, n = N_fft)

        H.append(H_i)
        X.append(S_i)

    E = sum([np.conj(H[i]) * X[i] for i in range(len(s))]) / sum([np.abs(H[i]) ** 2 for i in range(len(s))])
    e = np.real(np.fft.ifft(E))[:e_length]
    return e