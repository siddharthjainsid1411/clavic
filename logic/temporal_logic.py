# logic/temporal_logic.py

import numpy as np


def smooth_min(values, k=40.0):
    values = np.array(values)
    v_max = np.max(values)                            # shift to prevent underflow
    return v_max - 1.0 / k * np.log(np.sum(np.exp(-k * (values - v_max))))


def smooth_max(values, k=40.0):
    values = np.array(values)
    v_max = np.max(values)                            # shift to prevent overflow/underflow
    weights = np.exp(k * (values - v_max))
    return np.sum(values * weights) / (np.sum(weights) + 1e-12)


def eventually(rho, k=20.0):
    return smooth_max(rho, k)


def always(rho, k=20.0):
    return smooth_min(rho, k)


def until(rho_phi, rho_psi, k1=20.0, k2=20.0):
    T = len(rho_phi)
    values = []

    for t_prime in range(T):
        if t_prime == 0:
            min_before = np.inf
        else:
            min_before = smooth_min(rho_phi[:t_prime], k=k1)

        inner = smooth_min([rho_psi[t_prime], min_before], k=k1)
        values.append(inner)

    return smooth_max(values, k=k2)