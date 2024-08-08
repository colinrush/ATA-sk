import numpy as np
from guppi import guppi
import matplotlib.pyplot as plt

g = guppi.Guppi("/mnt/primary/scratch/crush/GUPPI/guppi_60516_74066_177093_J0332+5434_0001.0000.raw")

hdr, data = g.read_next_block()
NCHUNK = 1024
i_chunk = 0

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

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
while i_chunk * NCHUNK < data.shape[2]:
    d = data[:,:,i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,:]

    sk = calculate_sk(d)
    locMax = np.unravel_index(np.argmax(sk), sk.shape)
    locMin = np.unravel_index(np.argmin(sk), sk.shape)
    dataHighSK = data[locMax[0],locMax[1],i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,locMax[2]].real
    dataLowSK = data[locMin[0],locMin[1],i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,locMin[2]].real
    min_edge = min(dataLowSK.min(), dataHighSK.min())
    max_edge = max(dataLowSK.max(), dataHighSK.max())
    bins = np.linspace(min_edge, max_edge, 30) 
    # ,label="SK " + str(np.max))
    plt.hist(dataHighSK,bins=bins,alpha=0.5,label="Data with RFI, SK " + str(np.max(sk)))
    plt.hist(dataLowSK,bins=bins,alpha=0.5,label="No RFI, SK " + str(np.min(sk)))
    plt.xlabel("Intensity")
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.title("SK of Single-Channel Samples in ATA Data")
    plt.legend()
    plt.savefig("./SK_example_plots/lowSKlog" + str(i_chunk) + ".png")

    plt.close()

    i_chunk += 1

dist1 = np.random.normal(0,16,size=(2,1024))
dist2 = np.random.normal(0,16,size=(2,1024))
d_dt = dist1**2 + dist2**2
m=1024
s1 = d_dt.sum(axis=1)
print(s1)
print(s1**2)
s2 = (d_dt**2).sum(axis=1)
print(s2)
sk_array = (m + 1.) / (m - 1.) * (m*s2/s1**2 - 1)
print(sk_array)

d1 = dist1[0,:]
d2 = dist1[1,:]

min_edge = min(d1.min(), d2.min())
max_edge = max(d1.max(), d2.max())
bins = np.linspace(min_edge, max_edge, 30) 
# ,label="SK " + str(np.max))
plt.hist(d1,bins=bins,alpha=0.5,label="SK " + str(sk_array[0]))
plt.hist(d2,bins=bins,alpha=0.5,label="SK " + str(sk_array[1]))
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.title("SK of Normal Distributions")
plt.legend()
plt.yscale("log")
plt.savefig("./SK_example_plots/normallog" + str(i_chunk) + ".png")

plt.close()