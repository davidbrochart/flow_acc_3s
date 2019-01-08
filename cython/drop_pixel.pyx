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
def drop_pixel(np.ndarray[np.uint8_t, ndim=2] flow_dir, np.ndarray[np.float64_t, ndim=2] flow_acc, np.ndarray[np.float64_t, ndim=2] udlr_in, np.ndarray[np.float64_t, ndim=2] udlr_out, np.ndarray[np.float64_t, ndim=1] pix_area, bool do_inside, int row_i, int row_nb):
    cdef int y_nb, x_nb
    cdef int y0, x0, y, x
    cdef int do_it
    cdef double nb
    cdef int done
    cdef int dire
    cdef int do_inside_i
    do_inside_i = int(do_inside)
    y_nb = flow_dir.shape[0]
    x_nb = flow_dir.shape[1]
    with nogil:
        for y0 in range(row_i, row_i+row_nb):
            for x0 in range(x_nb):
                do_it = 1
                if y0 == 0:
                    # comes from tile above
                    nb = udlr_in[0, x0]
                elif y0 == y_nb - 1:
                    # comes from tile below
                    nb = udlr_in[1, x0]
                elif x0 == 0:
                    # comes from left tile
                    nb = udlr_in[2, y0]
                elif x0 == x_nb - 1:
                    # comes from right tile
                    nb = udlr_in[3, y0]
                else:
                    # inside tile
                    nb = 0
                    do_it = do_inside_i
                if (flow_acc[y0, x0] > 0.) and (nb == 0.):
                    do_it = 0
                if do_it == 1:
                    y, x = y0, x0
                    done = 0
                    while done == 0:
                        if flow_acc[y, x] == 0.:
                            nb += pix_area[y]
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
                            if dire == 255:
                                flow_acc[y, x] = 0.
                            done = 1
                        if y == -1:
                            udlr_out[0, x+1] += nb
                            done = 1
                        elif y == y_nb:
                            udlr_out[1, x+1] += nb
                            done = 1
                        elif x == -1:
                            udlr_out[2, y+1] += nb
                            done = 1
                        elif x == x_nb:
                            udlr_out[3, y+1] += nb
                            done = 1
