import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

np.import_array()

from libcpp cimport bool
from cpython cimport bool

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def drop_pixel(np.ndarray[np.uint8_t, ndim=2] flow_dir, np.ndarray[np.uint32_t, ndim=2] flow_acc, np.ndarray[np.uint32_t, ndim=2] udlr_in, np.ndarray[np.uint32_t, ndim=2] udlr_out, bool do_inside, int row_i):
    cdef int row_nb, col_nb
    cdef int y0, x0, y, x
    cdef bool do_it
    cdef int nb
    cdef bool done
    cdef int dire
    row_nb = flow_dir.shape[0]
    col_nb = flow_dir.shape[1]
    y0 = row_i
    #for y0 in range(row_nb):
    for x0 in range(col_nb):
        do_it = True
        if y0 == 0:
            # comes from tile above
            nb = udlr_in[0, x0]
        elif y0 == row_nb - 1:
            # comes from tile below
            nb = udlr_in[1, x0]
        elif x0 == 0:
            # comes from left tile
            nb = udlr_in[2, y0]
        elif x0 == col_nb - 1:
            # comes from right tile
            nb = udlr_in[3, y0]
        else:
            # inside tile
            nb = 0
            do_it = do_inside
        if (flow_acc[y0, x0] != 0) and (nb == 0):
            do_it = False
        if do_it:
            y, x = y0, x0
            done = False
            while not done:
                if flow_acc[y, x] == 0:
                    nb += 1
                flow_acc[y, x] += nb
                dire = flow_dir[y, x]
                if dire == 1:
                    x += 1
                elif dire == 2:
                    y += 1
                    x += 1
                elif dire == 4:
                    y += 1
                elif dire == 8:
                    y += 1
                    x -= 1
                elif dire == 16:
                    x -= 1
                elif dire == 32:
                    y -= 1
                    x -= 1
                elif dire == 64:
                    y -= 1
                elif dire == 128:
                    y -= 1
                    x += 1
                else:
                    done = True
                if y == -1:
                    udlr_out[0, x+1] += 1
                    done = True
                elif y == row_nb:
                    udlr_out[1, x+1] += 1
                    done = True
                elif x == -1:
                    udlr_out[2, y+1] += 1
                    done = True
                elif x == col_nb:
                    udlr_out[3, y+1] += 1
                    done = True
