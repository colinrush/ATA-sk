from guppi import guppi
from sigpyproc.readers import FilReader
from SKutils import calculate_sk, replace_values
import sys
from prep_filterbank import init_filterbank
import os
import numpy as np

# Used for generating filterbank output file header. Could be any filterbank file
FIL_COPY = "/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/fout.fil"

# Integration time and SK calculation sample size
NCHUNK = 1024
T_INT = 16

# For normal distribution replacement
std = 16
REPLACE_MEAN = 0

# Command line arguments
OUTFILE_NAME = sys.argv[1]
INFILE_DIR = sys.argv[2]
REPLACE_TYPE = sys.argv[3]

replace_dist = 0
print('Generating distribution, mean = 0, std =',std,'...')
real = np.random.normal(loc=0,scale=std,size=(100000,1024))
imag = np.random.normal(loc=0,scale=std,size=(100000,1024))
replace_dist = real + 1j*imag
print('Done')

def detect_integrate(data):
    detect = data.real*data.real + data.imag*data.imag
    dd = detect.sum(axis=-1)
    # integrate
    t = dd.reshape(dd.shape[0], -1, T_INT).sum(axis=-1)
    return t.T

gfiles = os.listdir(INFILE_DIR)
gfiles.sort()
fil = init_filterbank(FIL_COPY, INFILE_DIR + gfiles[0], OUTFILE_NAME, T_INT)
fileCount = 0
for fname in gfiles:
    g = guppi.Guppi(INFILE_DIR + fname)
    print(fname)
    block = 0
    while True:
        print('New block',fname,block,REPLACE_TYPE)

        block += 1
        hdr, data = g.read_next_block()
        if not hdr:
            break
        
        i_chunk = 0

        while i_chunk * NCHUNK < data.shape[2]:
            d = data[:,:,i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,:]
            #print(d)
            if REPLACE_TYPE != "none":
                sk = calculate_sk(d)
                d = replace_values(d, sk, 0.834186, 1.21695, replace_dist, value=REPLACE_TYPE)
            #print('after',d)
            bf = d.sum(axis = 0)
            
            di = detect_integrate(bf)
            di = di.astype(np.float32)

            di.tofile(fil)

            i_chunk += 1
    fileCount += 1


