[HydroSHEDS](https://www.hydrosheds.org) provides flow directions at 3 seconds
resolution, but flow accumulation is only available at 30s and 15s resolutions.
This is an attempt to fill that gap.

The computation takes place in two passes:
- first, pixels are dropped on each tile individually, and we keep track of the
  pixels that flow into neighbors. This pass can be parallelized and is the
most time consuming.
- second, we take those overflowing pixels and make them flow again. Note that
  this pass is not parallelized (except for the file compression), but it's
quite fast anyway because pixels only flow from the boundaries of the tiles (the
inside of the tiles doesn't have to be processed again).

Usually, flow accumulation is just pixel-based: when one pixel flows into
another, we just increment the flow. This doesn't take into account the area of
a pixel, but if we want to use flow accumulation for hydrologic purposes, we
must correct for the fact that a pixel at the equator is bigger than a pixel at
e.g. latitude 60N (since WGS84 is not an equal area projection). Here we do
this correction, and so the resulting flow accumulation data cannot be of
integer type (it is a `float64`).

If you want to run the Cython version (which is slightly faster), you first need
to compile it:

```
cd cython
python setup.py build_ext --inplace
```

For the Numba version, pass the `-n` flag. You can set the number of CPU cores
you want to use in each pass with e.g. `-p1 4 -p2 4`. Since it takes a lot of
time to compute (days, depending on your CPUs), the current state is saved after
each computation of a tile, so you can press Ctrl-C and resume later. But if you
want to start over, you can do so with the `-r` flag.

```
python flowAcc3s.py -n -r -p1 4 -p2 4
```
