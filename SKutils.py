import numpy as np

def calculate_sk(data):
    """
    data : np.ndarray-like
        data of shape (nants, nfreqs, nsamples, npols)
    """

    nants, nfreqs, nsamples, npols = data.shape
    m = nsamples
    #print(data)
    d_dt = data.real * data.real + data.imag * data.imag
    #print(d_dt)
    s1 = d_dt.sum(axis=2)
    s2 = (d_dt**2).sum(axis=2)
    sk_array = (m + 1.) / (m - 1.) * (m*s2/s1**2 - 1)
    #print(np.mean(sk_array))

    return sk_array


def replace_values(data, sk_array, sk_low, sk_upp, replace_dist,
                   value="median"):
    replace_index = int(np.random.random() * 100000)
    mean_real = 0
    std_real = 8
    mean_imag = 0
    std_imag = 8    
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

    if value not in ["zero", "median","mad","none", "normal","nearest"]:
        raise RuntimeError("can only use 'zero' or 'median' in value")

    for iant in range(nants):
        if value == "zero":
            to_replace = 0.
        elif value == "median":
            med_real = np.median(data[iant].real)
            med_imag = np.median(data[iant].imag)
            med = med_real + 1j*med_imag
            to_replace = med
            print('Replacing med_real',med_real,'med_imag',med_imag,'to_replace',to_replace)
        elif value == "mad":
            med = np.median(data[iant])
            abs_dev = np.abs(data[iant] - med)
            mad = np.median(abs_dev)
            to_replace = mad
        #elif value == "normal":
        #    to_replace = replace_dist[replace_index,:]
        #    replace_index = (replace_index + 1) % 1000000
        #    print(replace_index)
        

        for ifreq in range(nfreqs):
            for ipol in range(npols):
                sk_bool = (sk_array[iant, ifreq, ipol] < sk_low) or\
                          (sk_array[iant, ifreq, ipol] > sk_upp)
                
                if value == "nearest":
                    if not sk_bool:
                        if sk_array[iant,ifreq,ipol] < 1:
                            mean_real = np.mean(data[iant, ifreq, :, ipol].real)
                            std_real = np.std(data[iant,ifreq,:,ipol].real)
                            mean_imag = np.mean(data[iant, ifreq, :, ipol].imag)
                            std_imag = np.std(data[iant,ifreq,:,ipol].imag)

                if sk_bool:
                    if value == "normal":
                        to_replace = replace_dist[replace_index,:]
                        replace_index = (replace_index + 1) % 100000
                    if value == "nearest":
                        to_replace_real = np.random.normal(loc=mean_real,scale=std_real,size=(1024))
                        to_replace_imag = np.random.normal(loc=mean_imag,scale=std_imag,size=(1024))
                        to_replace = to_replace_real + 1j*to_replace_imag
                    data[iant, ifreq, :, ipol] = to_replace
    return(data)
