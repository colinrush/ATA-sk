import numpy as np

def calculate_sk(data):
    """
    data : np.ndarray-like
        data of shape (nants, nfreqs, nsamples, npols)
    """

    nants, nfreqs, nsamples, npols = data.shape
    m = nsamples

    d_dt = data.real * data.real + data.imag * data.imag
    s1 = d_dt.sum(axis=2)
    s2 = (d_dt**2).sum(axis=2)
    sk_array = (m + 1.) / (m - 1.) * (m*s2/s2**2 - 1)

    return sk_array


def replace_values(data, sk_array, sk_low, sk_upp,
                   value="median"):
    """
    In-place replacement of data 

    data : np.ndarray
        initial data of shape (nants, nfreqs, nsamps, npols)
    sk_array : np.ndarray
        SK values of shape (nants, nfreqs, npols)
    sk_low : float
        Low bound threshold for SK trigger
    sk_upp : float
        Upper bound threshold for SK trigger
    value : str
        either 'zero' or 'median'
    """
    nants, nfreqs, nsamples, npols = data.shape

    if value not in ["zero", "median"]:
        raise RuntimeError("can only use 'zero' or 'median' in value")

    for iant in range(nants):
        if value == "zero":
            to_replace = 0.
        elif value == "median":
            med_real = np.median(data[iant].real)
            med_imag = np.median(data[iant].imag)
            med = med_real + med_imag
            to_replace = med

        for ifreq in range(nfreqs):
            for ipol in range(npols):
                sk_bool = (sk_array[iant, ifreq, ipol] < sk_low) or\
                          (sk_array[iant, ifreq, ipol] > sk_upp)

                if sk_bool:
                    data[iant, ifreq, :, ipol] = to_replace
