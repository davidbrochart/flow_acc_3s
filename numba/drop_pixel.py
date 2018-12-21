from numba import jit

@jit(nopython=True, nogil=True)
def drop_pixel(flow_dir, flow_acc, udlr_in, udlr_out, do_inside, row_i, row_nb):
    y_nb, x_nb = flow_dir.shape
    for y0 in range(row_i, row_i+row_nb):
        for x0 in range(x_nb):
            do_it = True
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
                        udlr_out[0, x+1] += nb
                        done = True
                    elif y == y_nb:
                        udlr_out[1, x+1] += nb
                        done = True
                    elif x == -1:
                        udlr_out[2, y+1] += nb
                        done = True
                    elif x == x_nb:
                        udlr_out[3, y+1] += nb
                        done = True
