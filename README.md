[HydroSHEDS](https://www.hydrosheds.org) provides flow directions at 3sec
resolution, but flow accumulation is only available at 30s and 15s resolutions.
This is an attempt to fill that gap.

The computation takes place in two passes:
- first, pixels are dropped on each tile individually, and we keep track of the
  pixels that flow into neighbors. This pass can easily by parallelized and is
the most time consuming.
- second, we take those overflowing pixels and make them flow again. Note that
  this pass is not parallelized, but it's quite fast anyway because pixels flow
from the boundaries of the tiles only.

If you want to run the Cython version (which is slightly faster), you first need
to compile it:

```
cd cython
python setup.py build_ext --inplace
cd ..
python flowAcc3s.py
```

For the Numba version, pass the `-n` flag. You can set the number of CPU cores
you want to use with e.g. `-p 4`. Since it takes a lot of time to compute (days,
depending on your CPUs), the current state is saved after each computation of a
tile, so you can press Ctrl-C and resume later. But if you want to start over,
you can do so with the `-r` flag.

```
python flowAcc3s.py -n -r -p 4
```
