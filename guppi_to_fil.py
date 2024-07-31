from guppi import guppi
from sigpyproc.readers import FilReader
from SKutils import calculate_sk, replace_values
import sys
from prep_filterbank import prep_filterbank

FIL_COPY = "/mnt/primary/scratch/crush/guppi_60479_79539_018100_J0332+5434_0001/fout.fil"
NCHUNK = 2048
T_INT = 16

def detect_integrate(data):
    detect = data.real*data.real + data.imag*data.imag
    dd = detect.sum(axis=-1)
    # integrate
    t = dd.reshape(dd.shape[0], -1, T_INT).sum(axis=-1)
    return t.T

fil = prep_filterbank(FIL_COPY, sys.argv[2], sys.argv[1], T_INT)
for fname in sys.argv[2:]:
    g = guppi.Guppi(fname)

    while True:
        hdr, data = g.read_next_block()
        if not hdr:
            break

        i_chunk = 0

        while i_chunk * NCHUNK < data.shape[2]:
            d = data[:,:,i_chunk*NCHUNK:(i_chunk+1)*NCHUNK,:]
            sk = calculate_sk(d)
            d = replace_values(d, sk, 0.8,1.2)

            bf = d.sum(axis = 0)
            
            detect_integrate(bf).tofile(fil)




