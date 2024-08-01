from guppi import guppi
from sigpyproc.readers import FilReader
from SKutils import calculate_sk, replace_values
import sys
from prep_filterbank import init_filterbank
import os

FIL_COPY = "/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/fout.fil"
NCHUNK = 2048
T_INT = 16

def detect_integrate(data):
    detect = data.real*data.real + data.imag*data.imag
    dd = detect.sum(axis=-1)
    # integrate
    t = dd.reshape(dd.shape[0], -1, T_INT).sum(axis=-1)
    return t.T

dir = sys.argv[2]
gfiles = os.listdir(sys.argv[2])
gfiles.sort()
fil = init_filterbank(FIL_COPY, dir + gfiles[0], sys.argv[1], T_INT)
for fname in gfiles:
    g = guppi.Guppi(dir + fname)
    print(fname)
    block = 0
    while True:
        print('New block',fname,block,"m")
        block += 1
        hdr, data = g.read_next_block()
        if not hdr:
            break
        
        i_chunk = 0

        while i_chunk * NCHUNK < data.shape[2]:
            d = data[:,:,i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,:]
            sk = calculate_sk(d)
            d = replace_values(d, sk, 0.8,1.2, value="median")

            bf = d.sum(axis = 0)
            
            detect_integrate(bf).tofile(fil)

            i_chunk += NCHUNK




