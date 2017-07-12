# https://lists.mcs.anl.gov/pipermail/petsc-users/2010-February/005935.html
# https://lists.mcs.anl.gov/pipermail/petsc-users/attachments/20100218/cc29a146/attachment.py
# http://www.mcs.anl.gov/petsc/petsc-3.1/include/petscmat.h.html

import numpy as np
import os
import sys

COOKIE     = 1211216 # from petscmat.h
IntType    = '>i4'   # big-endian, 4 byte integer
ScalarType = '>f8'   # big-endian, 8 byte real floating

def readmat(filename):
    fh = open(filename, 'rb')
    try:
        header = np.fromfile(fh, dtype=IntType, count=4)
        assert header[0] == COOKIE 
        M, N, nz = header[1 :]
        #
        I = np.empty(M + 1, dtype=IntType)
        I[0] = 0 
        rownz = np.fromfile(fh, dtype=IntType, count=M)
        np.cumsum(rownz, out=I[1 :])
        assert I[-1] == nz
        #
        J = np.fromfile(fh, dtype=IntType,    count=nz)
        V = np.fromfile(fh, dtype=ScalarType, count=nz)
    finally:
        fh.close()
    #
    return (M, N), (I, J, V)

if __name__ == '__main__':
    input_file = sys.argv[1]
    (M, N), (I, J, V) = readmat(input_file)
    #
    file_name, file_extension = os.path.splitext(input_file)
    #
    new_file_extension = ".txt"
    new_file_name = file_name + new_file_extension
    #
    with open(new_file_name, "w") as ascii_file:
        for i in xrange(len(I) - 1):
            start, end = I[i], I[i + 1]
            colidx = J[start : end]
            values = V[start : end]
            str_to_print = 'row %d:' % i + str(zip(colidx, values))
            ascii_file.write("%s\n" % str_to_print)
