######
# Converts a guppi file to a filterbank file, replacing RFI with background noise. Channels are determined to
# be RFI using spectral kurtosis

from sigpyproc.readers import FilReader
import sys
import numpy as np
from astropy.time import Time
from guppi import guppi
import os

# TODO - take as arguments
path_to_guppi_directory = "/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/LoA.C0736/"
path_to_output = "./fout_0736_noreplace.fil"

## Guppi file parameters
NANT = 20 # Number of antennae in guppi file
NFREQ = 192 # Number of freq channels

## Spectral Kurtosis parameters
M = 2866 # Number of time intervals to calculate spectral kurtosis over (see Nita & Gary)
SK_LOWER = 0.865223 # Thresholds outside which to replace with random noise
SK_UPPER = 1.16982 

## Filterbank file paramets
T_INT = 16
# Filterbank file to copy the header from, see init_filterbank() function below
FIL_COPY = "/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/fout.fil" 

###
# A function to take a guppi data block (the data output of read_next_block()) and splits it into
# chunks of size chunk_size along the 3rd axis (time axis). Returns each chunk of the data in a list.
# If data.shape[2] % chunksize != 0, throws away the remaining time intervals. 
#
# Useful for setting M value for spectral kurtosis calculation, see Nita & Gary.
#  
# Inputs:
#   data - a 4-dimensional guppi block
#   chunk_size - an integer of desired final time interval length
# Output:
#   a list of 4-d guppi blocks that is the original data split into data.shape[2]/chunk_size pieces along
#   the time axis
def time_chunk_data(data, chunk_size):
    data_list = []
    data_length = data.shape[2]
    ind = 0
    # Iterates up to the end of the chunk size
    while ind + chunk_size <= data_length:
        data_list.append(data[:,:,ind:ind+chunk_size,:])
        ind += chunk_size
    return data_list

###
# A function to initialize the output filterbank file. This preps the header by copying a 'junk' filterbank
# file header and updating it with the headerfile from the input guppi file.
#
# Input:
#   - path_to_fil: path to any filterbank file, to copy the header
#   - path_to_guppi: path to input guppi file to be converted to filterbank
def init_filterbank(path_to_fil, path_to_guppi, outfile_name):
    # fil = FilReader("/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/fout.fil")

    # Opens filterbank file for example header
    fil = FilReader(path_to_fil)
    header = fil.header

    #dir = "./LoA.C0544/"
    # gfiles = os.listdir("/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/LoA.C0544")
    
    # Opens guppi file to populate filterbank header
    gup_file = guppi.Guppi(path_to_guppi)
    hdr, block = gup_file.read_next_block()

    # Copy info from guppi header to filterbank header
    header.fch1 = float(hdr['OBSFREQ']) - hdr['OBSBW']/2.0
    header.foff = float(hdr['CHAN_BW'])
    header.nchans = hdr['NCHAN']
    header.source = hdr['SOURCE']
    header.nbits = 32
    # header.bandwidth = float(hdr['OBSBW'])
    tstart_unix = hdr['PKTSTART']*float(hdr['TBIN']) + hdr['SYNCTIME']
    t = Time(tstart_unix, format='unix')
    header.tstart = t.mjd
    header.tsamp = hdr['TBIN']*T_INT

    # some weird sorcery, leave as is
    fout_name = outfile_name
    fio = header.prep_outfile(fout_name)
    fio.close()

    # Returns pointer to open filterbank file
    return open(fout_name, "ab")

##
# Spectral Kurtosis
def sk_replace(data):
    for iant in range(NANT):
        for ifreq in range(NFREQ):
            # get the data for that freq channel
            d = data[iant, ifreq, :, 0]
            # detect data
            d_dt = d.real * d.real + d.imag * d.imag

            # Calculate statistics
            s1 = d_dt.sum()
            s2 = (d_dt**2).sum()
            #print(M)
            # The Spectral Kurtosis
            sk = (M + 1.)/(M - 1) * (M*s2/s1**2 - 1)

            sk_bool = (sk < SK_LOWER) or (sk > SK_UPPER)

            if sk_bool:
                real = np.random.normal(loc=0,scale=14.4,size=(1,1,M,2))
                whole = real + 1j * real
                data[iant, ifreq, :, :] = whole
    return data

# Get the guppi files to combine
gfiles = os.listdir(path_to_guppi_directory)
gfiles.sort()

# Printing for sanity, ensure correct order:
print("Combining")
for i in range(len(gfiles)):
    print(gfiles[i])

fio = init_filterbank(FIL_COPY, path_to_guppi_directory + "/" + gfiles[0], path_to_output)

for i in range(len(gfiles)):
    print(gfiles[i])
    g = guppi.Guppi(path_to_guppi_directory + gfiles[i])
    j = 0
    while True:
        print('NEW BLOCK', j)
        j += 1
        hdr, full_block = g.read_next_block()
        if not hdr:
            break
        datalist = time_chunk_data(full_block, M)
        result_list = []

        for h in range(len(datalist)):
            
            data = datalist[h]

            result_list.append(sk_replace(data))
            
        data = np.concatenate(datalist,axis=2)
        
        data = data.sum(axis=0)
        detect = data.real*data.real + data.imag*data.imag
        dd = detect.sum(axis=-1)
        # integrate
        t = dd.reshape(NFREQ, -1, T_INT).sum(axis=-1)
        # write out
        t.T.tofile(fio)



# fil = FilReader("/mnt/primary/ata/projects/p031/fil_60480_46898_2800170_frb20240114a_0001/LoA.C0544/fil_60480_46898_2800170_frb20240114a_0001-beam0000.fil")
