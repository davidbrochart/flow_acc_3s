[HydroSHEDS](https://www.hydrosheds.org) provides flow directions at 3sec
resolution, but flow accumulation is only available at 30s and 15s resolutions.
This is an attempt to fill that gap.

If you want to run the Cython version (which is slightly faster), you first need
to compile it:

```
cd cython
python setup.py build_ext --inplace
cd ..
python flowAcc3s.py
```

For the Numba version, just run:

```
python flowAcc3s.py -n
```

Since it takes a lot of time to compute (days), the current state is saved after
each computation of a tile. You can e.g. hit Ctrl-C and resume later. But if you
want to start over, you first need to:

```
rm -rf tiles/acc tmp
```
