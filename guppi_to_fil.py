from guppi import guppi
from SKutils import calculate_sk, replace_values
import sys

NCHUNK = 2048

for fname in sys.argv[1:]:
    g = guppi.Guppi(fname)

    hdr, data = g.read_next_block()

    for ichunk in range(4):
        data_chunk = data[:, :, i*NCHUNK:(i+1)*NCHUNK, :]
        sk_array_chunk = calculate_sk(data_chunk)

    detect_integrate()
