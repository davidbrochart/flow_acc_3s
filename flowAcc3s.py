import os
import sys
import zipfile
import requests
import shutil
from osgeo import gdal
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import click

def get_flow_dir(row, dir_path):
    if not os.path.exists(dir_path + row['tile']):
        print('Downloading ' + row['tile'] + '...')
        r = requests.get(row['url'] + row['tile'])
        with open(dir_path + row['tile'], 'wb') as f:
            f.write(r.content)
    try:
        with zipfile.ZipFile(dir_path + row['tile'], 'r') as z:
            z.extractall(path = 'tmp/')
        flow_dir = gdal.Open('tmp/' + row['tile'][:-9] + '/' + row['tile'][:-9] + '/w001001.adf')
        geo = flow_dir.GetGeoTransform()
        ySize, xSize = flow_dir.RasterYSize, flow_dir.RasterXSize
        flow_dir = flow_dir.ReadAsArray()
        shutil.rmtree('tmp/' + row['tile'][:-9])
        # data is padded into a 6000x6000 array (some tiles may be smaller):
        array_5x5 = np.zeros((6000, 6000), dtype = 'uint8')
        y0 = int(round((geo[3] - row.lat) / geo[5]))
        y1 = 6000 - int(round(((row.lat - 5) - (geo[3] + geo[5] * ySize)) / geo[5]))
        x0 = int(round((geo[0] - row.lon) / geo[1]))
        x1 = 6000 - int(round(((row.lon + 5) - (geo[0] + geo[1] * xSize)) / geo[1]))
        array_5x5[y0:y1, x0:x1] = flow_dir
    except:
        print('Not a ZIP file!')
        array_5x5 = np.zeros((6000, 6000), dtype = 'uint8')
    return array_5x5

@click.command()
@click.option('-n', '--numba', is_flag=True, help='Use Numba as the computing backend (otherwise, use Cython).')
def acc_flow(numba):
    if numba:
        sys.path.append('numba')
        print('Using Numba')
    else:
        sys.path.append('cython')
        print('Using Cython')
    from drop_pixel import drop_pixel
    dir_path = 'tiles/dir/3s/'
    acc_path = 'tiles/acc/3s/'
    udlr_path = 'tmp/udlr'
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(udlr_path, exist_ok=True)
    
    try:
        df = pd.read_pickle('tmp/df.pkl')
    except:
        dire = {
            'Africa': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/AF/', ['n00e005_dir_grid.zip', 'n00e010_dir_grid.zip', 'n00e015_dir_grid.zip', 'n00e020_dir_grid.zip', 'n00e025_dir_grid.zip', 'n00e030_dir_grid.zip', 'n00e035_dir_grid.zip', 'n00e040_dir_grid.zip', 'n00e045_dir_grid.zip', 'n00w005_dir_grid.zip', 'n00w010_dir_grid.zip', 'n05e000_dir_grid.zip', 'n05e005_dir_grid.zip', 'n05e010_dir_grid.zip', 'n05e015_dir_grid.zip', 'n05e020_dir_grid.zip', 'n05e025_dir_grid.zip', 'n05e030_dir_grid.zip', 'n05e035_dir_grid.zip', 'n05e040_dir_grid.zip', 'n05e045_dir_grid.zip', 'n05e050_dir_grid.zip', 'n05w005_dir_grid.zip', 'n05w005_dir_grid.zip', 'n05w010_dir_grid.zip', 'n10e000_dir_grid.zip', 'n10e005_dir_grid.zip', 'n10e010_dir_grid.zip', 'n10e015_dir_grid.zip', 'n10e020_dir_grid.zip', 'n10e025_dir_grid.zip', 'n10e030_dir_grid.zip', 'n10e035_dir_grid.zip', 'n10e040_dir_grid.zip', 'n10e045_dir_grid.zip', 'n10e050_dir_grid.zip', 'n10w005_dir_grid.zip', 'n10w010_dir_grid.zip', 'n10w015_dir_grid.zip', 'n10w020_dir_grid.zip', 'n15e000_dir_grid.zip', 'n15e005_dir_grid.zip', 'n15e010_dir_grid.zip', 'n15e015_dir_grid.zip', 'n15e020_dir_grid.zip', 'n15e025_dir_grid.zip', 'n15e030_dir_grid.zip', 'n15e035_dir_grid.zip', 'n15e040_dir_grid.zip', 'n15e045_dir_grid.zip', 'n15e050_dir_grid.zip', 'n15e055_dir_grid.zip', 'n15w005_dir_grid.zip', 'n15w010_dir_grid.zip', 'n15w015_dir_grid.zip', 'n15w020_dir_grid.zip', 'n20e000_dir_grid.zip', 'n20e005_dir_grid.zip', 'n20e010_dir_grid.zip', 'n20e015_dir_grid.zip', 'n20e020_dir_grid.zip', 'n20e025_dir_grid.zip', 'n20e030_dir_grid.zip', 'n20e035_dir_grid.zip', 'n20e040_dir_grid.zip', 'n20e045_dir_grid.zip', 'n20e050_dir_grid.zip', 'n20w005_dir_grid.zip', 'n20w010_dir_grid.zip', 'n20w015_dir_grid.zip', 'n20w020_dir_grid.zip', 'n25e000_dir_grid.zip', 'n25e005_dir_grid.zip', 'n25e010_dir_grid.zip', 'n25e015_dir_grid.zip', 'n25e020_dir_grid.zip', 'n25e025_dir_grid.zip', 'n25e030_dir_grid.zip', 'n25e035_dir_grid.zip', 'n25e040_dir_grid.zip', 'n25e045_dir_grid.zip', 'n25e050_dir_grid.zip', 'n25w005_dir_grid.zip', 'n25w010_dir_grid.zip', 'n25w015_dir_grid.zip', 'n25w020_dir_grid.zip', 'n30e000_dir_grid.zip', 'n30e005_dir_grid.zip', 'n30e010_dir_grid.zip', 'n30e015_dir_grid.zip', 'n30e020_dir_grid.zip', 'n30e025_dir_grid.zip', 'n30e030_dir_grid.zip', 'n30e035_dir_grid.zip', 'n30e040_dir_grid.zip', 'n30e045_dir_grid.zip', 'n30e050_dir_grid.zip', 'n30w005_dir_grid.zip', 'n30w010_dir_grid.zip', 'n30w020_dir_grid.zip', 'n35e000_dir_grid.zip', 'n35e005_dir_grid.zip', 'n35e010_dir_grid.zip', 'n35e015_dir_grid.zip', 'n35e020_dir_grid.zip', 'n35e025_dir_grid.zip', 'n35e030_dir_grid.zip', 'n35e035_dir_grid.zip', 'n35e040_dir_grid.zip', 'n35e045_dir_grid.zip', 'n35e050_dir_grid.zip', 'n35w005_dir_grid.zip', 'n35w010_dir_grid.zip', 's05e005_dir_grid.zip', 's05e010_dir_grid.zip', 's05e015_dir_grid.zip', 's05e020_dir_grid.zip', 's05e025_dir_grid.zip', 's05e030_dir_grid.zip', 's05e035_dir_grid.zip', 's05e040_dir_grid.zip', 's05e045_dir_grid.zip', 's05e050_dir_grid.zip', 's10e010_dir_grid.zip', 's10e015_dir_grid.zip', 's10e020_dir_grid.zip', 's10e025_dir_grid.zip', 's10e030_dir_grid.zip', 's10e035_dir_grid.zip', 's10e040_dir_grid.zip', 's10e045_dir_grid.zip', 's10e050_dir_grid.zip', 's15e010_dir_grid.zip', 's15e015_dir_grid.zip', 's15e020_dir_grid.zip', 's15e025_dir_grid.zip', 's15e030_dir_grid.zip', 's15e035_dir_grid.zip', 's15e040_dir_grid.zip', 's15e045_dir_grid.zip', 's15e050_dir_grid.zip', 's20e010_dir_grid.zip', 's20e015_dir_grid.zip', 's20e020_dir_grid.zip', 's20e025_dir_grid.zip', 's20e030_dir_grid.zip', 's20e035_dir_grid.zip', 's20e040_dir_grid.zip', 's20e045_dir_grid.zip', 's20e050_dir_grid.zip', 's25e010_dir_grid.zip', 's25e015_dir_grid.zip', 's25e020_dir_grid.zip', 's25e025_dir_grid.zip', 's25e030_dir_grid.zip', 's25e035_dir_grid.zip', 's25e040_dir_grid.zip', 's25e045_dir_grid.zip', 's30e010_dir_grid.zip', 's30e015_dir_grid.zip', 's30e020_dir_grid.zip', 's30e025_dir_grid.zip', 's30e030_dir_grid.zip', 's30e040_dir_grid.zip', 's30e045_dir_grid.zip', 's35e015_dir_grid.zip', 's35e020_dir_grid.zip', 's35e025_dir_grid.zip', 's35e030_dir_grid.zip']],
            'Asia': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/AS/', ['n00e095_dir_grid.zip', 'n00e100_dir_grid.zip', 'n00e105_dir_grid.zip', 'n00e110_dir_grid.zip', 'n00e115_dir_grid.zip', 'n00e120_dir_grid.zip', 'n00e125_dir_grid.zip', 'n00e130_dir_grid.zip', 'n00e150_dir_grid.zip', 'n00e155_dir_grid.zip', 'n00e165_dir_grid.zip', 'n00e170_dir_grid.zip', 'n05e075_dir_grid.zip', 'n05e080_dir_grid.zip', 'n05e090_dir_grid.zip', 'n05e095_dir_grid.zip', 'n05e100_dir_grid.zip', 'n05e105_dir_grid.zip', 'n05e110_dir_grid.zip', 'n05e115_dir_grid.zip', 'n05e120_dir_grid.zip', 'n05e125_dir_grid.zip', 'n05e130_dir_grid.zip', 'n05e135_dir_grid.zip', 'n05e140_dir_grid.zip', 'n05e145_dir_grid.zip', 'n05e150_dir_grid.zip', 'n05e155_dir_grid.zip', 'n05e160_dir_grid.zip', 'n05e165_dir_grid.zip', 'n05e170_dir_grid.zip', 'n10e070_dir_grid.zip', 'n10e075_dir_grid.zip', 'n10e080_dir_grid.zip', 'n10e090_dir_grid.zip', 'n10e095_dir_grid.zip', 'n10e100_dir_grid.zip', 'n10e105_dir_grid.zip', 'n10e110_dir_grid.zip', 'n10e115_dir_grid.zip', 'n10e120_dir_grid.zip', 'n10e125_dir_grid.zip', 'n10e135_dir_grid.zip', 'n10e140_dir_grid.zip', 'n10e145_dir_grid.zip', 'n10e160_dir_grid.zip', 'n10e165_dir_grid.zip', 'n10e170_dir_grid.zip', 'n15e055_dir_grid.zip', 'n15e070_dir_grid.zip', 'n15e075_dir_grid.zip', 'n15e080_dir_grid.zip', 'n15e085_dir_grid.zip', 'n15e090_dir_grid.zip', 'n15e095_dir_grid.zip', 'n15e100_dir_grid.zip', 'n15e105_dir_grid.zip', 'n15e110_dir_grid.zip', 'n15e115_dir_grid.zip', 'n15e120_dir_grid.zip', 'n15e145_dir_grid.zip', 'n15e165_dir_grid.zip', 'n20e055_dir_grid.zip', 'n20e065_dir_grid.zip', 'n20e070_dir_grid.zip', 'n20e075_dir_grid.zip', 'n20e080_dir_grid.zip', 'n20e085_dir_grid.zip', 'n20e090_dir_grid.zip', 'n20e095_dir_grid.zip', 'n20e100_dir_grid.zip', 'n20e105_dir_grid.zip', 'n20e110_dir_grid.zip', 'n20e115_dir_grid.zip', 'n20e120_dir_grid.zip', 'n20e125_dir_grid.zip', 'n20e130_dir_grid.zip', 'n20e135_dir_grid.zip', 'n20e140_dir_grid.zip', 'n20e145_dir_grid.zip', 'n20e150_dir_grid.zip', 'n25e055_dir_grid.zip', 'n25e060_dir_grid.zip', 'n25e065_dir_grid.zip', 'n25e070_dir_grid.zip', 'n25e075_dir_grid.zip', 'n25e080_dir_grid.zip', 'n25e085_dir_grid.zip', 'n25e090_dir_grid.zip', 'n25e095_dir_grid.zip', 'n25e100_dir_grid.zip', 'n25e105_dir_grid.zip', 'n25e110_dir_grid.zip', 'n25e115_dir_grid.zip', 'n25e120_dir_grid.zip', 'n25e125_dir_grid.zip', 'n25e130_dir_grid.zip', 'n25e140_dir_grid.zip', 'n30e055_dir_grid.zip', 'n30e060_dir_grid.zip', 'n30e065_dir_grid.zip', 'n30e070_dir_grid.zip', 'n30e075_dir_grid.zip', 'n30e080_dir_grid.zip', 'n30e085_dir_grid.zip', 'n30e090_dir_grid.zip', 'n30e095_dir_grid.zip', 'n30e100_dir_grid.zip', 'n30e105_dir_grid.zip', 'n30e110_dir_grid.zip', 'n30e115_dir_grid.zip', 'n30e120_dir_grid.zip', 'n30e125_dir_grid.zip', 'n30e130_dir_grid.zip', 'n30e140_dir_grid.zip', 'n35e055_dir_grid.zip', 'n35e060_dir_grid.zip', 'n35e065_dir_grid.zip', 'n35e070_dir_grid.zip', 'n35e075_dir_grid.zip', 'n35e080_dir_grid.zip', 'n35e085_dir_grid.zip', 'n35e090_dir_grid.zip', 'n35e095_dir_grid.zip', 'n35e100_dir_grid.zip', 'n35e105_dir_grid.zip', 'n35e110_dir_grid.zip', 'n35e115_dir_grid.zip', 'n35e120_dir_grid.zip', 'n35e125_dir_grid.zip', 'n35e130_dir_grid.zip', 'n35e140_dir_grid.zip', 'n40e055_dir_grid.zip', 'n40e060_dir_grid.zip', 'n40e065_dir_grid.zip', 'n40e070_dir_grid.zip', 'n40e075_dir_grid.zip', 'n40e080_dir_grid.zip', 'n40e085_dir_grid.zip', 'n40e090_dir_grid.zip', 'n40e095_dir_grid.zip', 'n40e100_dir_grid.zip', 'n40e105_dir_grid.zip', 'n40e110_dir_grid.zip', 'n40e115_dir_grid.zip', 'n40e120_dir_grid.zip', 'n40e125_dir_grid.zip', 'n40e130_dir_grid.zip', 'n40e135_dir_grid.zip', 'n40e140_dir_grid.zip', 'n40e145_dir_grid.zip', 'n45e055_dir_grid.zip', 'n45e060_dir_grid.zip', 'n45e065_dir_grid.zip', 'n45e070_dir_grid.zip', 'n45e075_dir_grid.zip', 'n45e080_dir_grid.zip', 'n45e085_dir_grid.zip', 'n45e090_dir_grid.zip', 'n45e095_dir_grid.zip', 'n45e100_dir_grid.zip', 'n45e105_dir_grid.zip', 'n45e110_dir_grid.zip', 'n45e115_dir_grid.zip', 'n45e120_dir_grid.zip', 'n45e125_dir_grid.zip', 'n45e130_dir_grid.zip', 'n45e135_dir_grid.zip', 'n45e140_dir_grid.zip', 'n45e145_dir_grid.zip', 'n45e150_dir_grid.zip', 'n45e155_dir_grid.zip', 'n50e055_dir_grid.zip', 'n50e060_dir_grid.zip', 'n50e065_dir_grid.zip', 'n50e070_dir_grid.zip', 'n50e075_dir_grid.zip', 'n50e080_dir_grid.zip', 'n50e085_dir_grid.zip', 'n50e090_dir_grid.zip', 'n50e095_dir_grid.zip', 'n50e100_dir_grid.zip', 'n50e105_dir_grid.zip', 'n50e110_dir_grid.zip', 'n50e115_dir_grid.zip', 'n50e120_dir_grid.zip', 'n50e125_dir_grid.zip', 'n50e130_dir_grid.zip', 'n50e135_dir_grid.zip', 'n50e140_dir_grid.zip', 'n50e150_dir_grid.zip', 'n50e155_dir_grid.zip', 'n50e160_dir_grid.zip', 'n50e165_dir_grid.zip', 'n50e170_dir_grid.zip', 'n50e175_dir_grid.zip', 'n55e055_dir_grid.zip', 'n55e060_dir_grid.zip', 'n55e065_dir_grid.zip', 'n55e070_dir_grid.zip', 'n55e075_dir_grid.zip', 'n55e080_dir_grid.zip', 'n55e085_dir_grid.zip', 'n55e090_dir_grid.zip', 'n55e095_dir_grid.zip', 'n55e100_dir_grid.zip', 'n55e105_dir_grid.zip', 'n55e110_dir_grid.zip', 'n55e115_dir_grid.zip', 'n55e120_dir_grid.zip', 'n55e125_dir_grid.zip', 'n55e130_dir_grid.zip', 'n55e135_dir_grid.zip', 'n55e140_dir_grid.zip', 'n55e145_dir_grid.zip', 'n55e150_dir_grid.zip', 'n55e155_dir_grid.zip', 'n55e160_dir_grid.zip', 'n55e165_dir_grid.zip', 'n55e170_dir_grid.zip', 's05e095_dir_grid.zip', 's05e100_dir_grid.zip', 's05e105_dir_grid.zip', 's05e110_dir_grid.zip', 's05e115_dir_grid.zip', 's05e120_dir_grid.zip', 's05e125_dir_grid.zip', 's05e130_dir_grid.zip', 's05e135_dir_grid.zip', 's05e140_dir_grid.zip', 's05e150_dir_grid.zip', 's05e155_dir_grid.zip', 's05e165_dir_grid.zip', 's05e170_dir_grid.zip', 's05e175_dir_grid.zip', 's10e100_dir_grid.zip', 's10e105_dir_grid.zip', 's10e110_dir_grid.zip', 's10e115_dir_grid.zip', 's10e120_dir_grid.zip', 's10e125_dir_grid.zip', 's10e130_dir_grid.zip', 's10e135_dir_grid.zip', 's10e140_dir_grid.zip', 's10e150_dir_grid.zip', 's10e155_dir_grid.zip', 's10e160_dir_grid.zip', 's10e165_dir_grid.zip', 's10e175_dir_grid.zip']],
            'Australia': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/au/', ['s05e110_dir_grid.zip', 's05e115_dir_grid.zip', 's05e120_dir_grid.zip', 's05e125_dir_grid.zip', 's05e130_dir_grid.zip', 's05e135_dir_grid.zip', 's05e140_dir_grid.zip', 's05e145_dir_grid.zip', 's05e150_dir_grid.zip', 's05e155_dir_grid.zip', 's05e165_dir_grid.zip', 's05e170_dir_grid.zip', 's05e175_dir_grid.zip', 's10e110_dir_grid.zip', 's10e115_dir_grid.zip', 's10e120_dir_grid.zip', 's10e125_dir_grid.zip', 's10e130_dir_grid.zip', 's10e135_dir_grid.zip', 's10e140_dir_grid.zip', 's10e145_dir_grid.zip', 's10e150_dir_grid.zip', 's10e155_dir_grid.zip', 's10e160_dir_grid.zip', 's10e165_dir_grid.zip', 's10e175_dir_grid.zip', 's15e115_dir_grid.zip', 's15e120_dir_grid.zip', 's15e125_dir_grid.zip', 's15e130_dir_grid.zip', 's15e135_dir_grid.zip', 's15e140_dir_grid.zip', 's15e145_dir_grid.zip', 's15e150_dir_grid.zip', 's15e155_dir_grid.zip', 's15e160_dir_grid.zip', 's15e165_dir_grid.zip', 's15e170_dir_grid.zip', 's15e175_dir_grid.zip', 's20e115_dir_grid.zip', 's20e120_dir_grid.zip', 's20e125_dir_grid.zip', 's20e130_dir_grid.zip', 's20e135_dir_grid.zip', 's20e140_dir_grid.zip', 's20e145_dir_grid.zip', 's20e150_dir_grid.zip', 's20e155_dir_grid.zip', 's20e160_dir_grid.zip', 's20e165_dir_grid.zip', 's20e170_dir_grid.zip', 's20e175_dir_grid.zip', 's25e110_dir_grid.zip', 's25e115_dir_grid.zip', 's25e120_dir_grid.zip', 's25e125_dir_grid.zip', 's25e130_dir_grid.zip', 's25e135_dir_grid.zip', 's25e140_dir_grid.zip', 's25e145_dir_grid.zip', 's25e150_dir_grid.zip', 's25e155_dir_grid.zip', 's25e160_dir_grid.zip', 's25e165_dir_grid.zip', 's25e170_dir_grid.zip', 's30e110_dir_grid.zip', 's30e115_dir_grid.zip', 's30e120_dir_grid.zip', 's30e125_dir_grid.zip', 's30e130_dir_grid.zip', 's30e135_dir_grid.zip', 's30e140_dir_grid.zip', 's30e145_dir_grid.zip', 's30e150_dir_grid.zip', 's30e165_dir_grid.zip', 's35e110_dir_grid.zip', 's35e115_dir_grid.zip', 's35e120_dir_grid.zip', 's35e125_dir_grid.zip', 's35e130_dir_grid.zip', 's35e135_dir_grid.zip', 's35e140_dir_grid.zip', 's35e145_dir_grid.zip', 's35e150_dir_grid.zip', 's35e155_dir_grid.zip', 's35e170_dir_grid.zip', 's40e115_dir_grid.zip', 's40e135_dir_grid.zip', 's40e140_dir_grid.zip', 's40e145_dir_grid.zip', 's40e150_dir_grid.zip', 's40e170_dir_grid.zip', 's40e175_dir_grid.zip', 's45e140_dir_grid.zip', 's45e145_dir_grid.zip', 's45e165_dir_grid.zip', 's45e170_dir_grid.zip', 's45e175_dir_grid.zip', 's50e165_dir_grid.zip', 's50e170_dir_grid.zip', 's50e175_dir_grid.zip', 's55e155_dir_grid.zip', 's55e165_dir_grid.zip', 's60e155_dir_grid.zip']],
            'Central America, Caribbean, Mexico': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/CA/', ['n05w060_dir_grid.zip', 'n05w065_dir_grid.zip', 'n05w070_dir_grid.zip', 'n05w075_dir_grid.zip', 'n05w080_dir_grid.zip', 'n05w085_dir_grid.zip', 'n05w090_dir_grid.zip', 'n10w060_dir_grid.zip', 'n10w065_dir_grid.zip', 'n10w070_dir_grid.zip', 'n10w075_dir_grid.zip', 'n10w080_dir_grid.zip', 'n10w085_dir_grid.zip', 'n10w090_dir_grid.zip', 'n10w095_dir_grid.zip', 'n10w110_dir_grid.zip', 'n15w065_dir_grid.zip', 'n15w070_dir_grid.zip', 'n15w075_dir_grid.zip', 'n15w080_dir_grid.zip', 'n15w085_dir_grid.zip', 'n15w090_dir_grid.zip', 'n15w095_dir_grid.zip', 'n15w100_dir_grid.zip', 'n15w105_dir_grid.zip', 'n15w110_dir_grid.zip', 'n15w115_dir_grid.zip', 'n20w075_dir_grid.zip', 'n20w080_dir_grid.zip', 'n20w085_dir_grid.zip', 'n20w090_dir_grid.zip', 'n20w095_dir_grid.zip', 'n20w100_dir_grid.zip', 'n20w110_dir_grid.zip', 'n20w115_dir_grid.zip', 'n20w120_dir_grid.zip', 'n25w080_dir_grid.zip', 'n25w085_dir_grid.zip', 'n25w090_dir_grid.zip', 'n25w095_dir_grid.zip', 'n25w100_dir_grid.zip', 'n25w110_dir_grid.zip', 'n25w115_dir_grid.zip', 'n25w120_dir_grid.zip', 'n30w080_dir_grid.zip', 'n30w085_dir_grid.zip', 'n30w090_dir_grid.zip', 'n30w095_dir_grid.zip', 'n30w100_dir_grid.zip', 'n30w110_dir_grid.zip', 'n30w115_dir_grid.zip', 'n30w120_dir_grid.zip', 'n30w125_dir_grid.zip', 'n35w075_dir_grid.zip', 'n35w080_dir_grid.zip', 'n35w085_dir_grid.zip', 'n35w090_dir_grid.zip', 'n35w095_dir_grid.zip', 'n35w100_dir_grid.zip', 'n35w110_dir_grid.zip', 'n35w115_dir_grid.zip', 'n35w120_dir_grid.zip', 'n35w125_dir_grid.zip']],
            'Europe, Southwest Asia': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/EU/', ['n10e000_dir_grid.zip', 'n10e005_dir_grid.zip', 'n10e010_dir_grid.zip', 'n10e015_dir_grid.zip', 'n10e020_dir_grid.zip', 'n10e025_dir_grid.zip', 'n10e030_dir_grid.zip', 'n10e035_dir_grid.zip', 'n10e040_dir_grid.zip', 'n10e045_dir_grid.zip', 'n10e050_dir_grid.zip', 'n10e070_dir_grid.zip', 'n10w005_dir_grid.zip', 'n10w010_dir_grid.zip', 'n10w015_dir_grid.zip', 'n10w020_dir_grid.zip', 'n15e000_dir_grid.zip', 'n15e005_dir_grid.zip', 'n15e010_dir_grid.zip', 'n15e015_dir_grid.zip', 'n15e020_dir_grid.zip', 'n15e025_dir_grid.zip', 'n15e030_dir_grid.zip', 'n15e035_dir_grid.zip', 'n15e040_dir_grid.zip', 'n15e045_dir_grid.zip', 'n15e050_dir_grid.zip', 'n15e055_dir_grid.zip', 'n15w005_dir_grid.zip', 'n15w010_dir_grid.zip', 'n15w015_dir_grid.zip', 'n15w020_dir_grid.zip', 'n20e000_dir_grid.zip', 'n20e005_dir_grid.zip', 'n20e010_dir_grid.zip', 'n20e015_dir_grid.zip', 'n20e020_dir_grid.zip', 'n20e025_dir_grid.zip', 'n20e030_dir_grid.zip', 'n20e035_dir_grid.zip', 'n20e040_dir_grid.zip', 'n20e045_dir_grid.zip', 'n20e050_dir_grid.zip', 'n20e055_dir_grid.zip', 'n20e065_dir_grid.zip', 'n20w005_dir_grid.zip', 'n20w010_dir_grid.zip', 'n20w015_dir_grid.zip', 'n20w020_dir_grid.zip', 'n25e000_dir_grid.zip', 'n25e005_dir_grid.zip', 'n25e010_dir_grid.zip', 'n25e015_dir_grid.zip', 'n25e020_dir_grid.zip', 'n25e025_dir_grid.zip', 'n25e030_dir_grid.zip', 'n25e035_dir_grid.zip', 'n25e040_dir_grid.zip', 'n25e045_dir_grid.zip', 'n25e050_dir_grid.zip', 'n25e055_dir_grid.zip', 'n25e060_dir_grid.zip', 'n25e065_dir_grid.zip', 'n25w005_dir_grid.zip', 'n25w010_dir_grid.zip', 'n25w015_dir_grid.zip', 'n25w020_dir_grid.zip', 'n30e000_dir_grid.zip', 'n30e005_dir_grid.zip', 'n30e010_dir_grid.zip', 'n30e015_dir_grid.zip', 'n30e020_dir_grid.zip', 'n30e025_dir_grid.zip', 'n30e030_dir_grid.zip', 'n30e035_dir_grid.zip', 'n30e040_dir_grid.zip', 'n30e045_dir_grid.zip', 'n30e050_dir_grid.zip', 'n30e055_dir_grid.zip', 'n30e060_dir_grid.zip', 'n30e065_dir_grid.zip', 'n30w005_dir_grid.zip', 'n30w010_dir_grid.zip', 'n30w020_dir_grid.zip', 'n35e000_dir_grid.zip', 'n35e005_dir_grid.zip', 'n35e010_dir_grid.zip', 'n35e015_dir_grid.zip', 'n35e020_dir_grid.zip', 'n35e025_dir_grid.zip', 'n35e030_dir_grid.zip', 'n35e035_dir_grid.zip', 'n35e040_dir_grid.zip', 'n35e045_dir_grid.zip', 'n35e050_dir_grid.zip', 'n35e055_dir_grid.zip', 'n35e060_dir_grid.zip', 'n35e065_dir_grid.zip', 'n35w005_dir_grid.zip', 'n35w010_dir_grid.zip', 'n40e000_dir_grid.zip', 'n40e005_dir_grid.zip', 'n40e010_dir_grid.zip', 'n40e015_dir_grid.zip', 'n40e020_dir_grid.zip', 'n40e025_dir_grid.zip', 'n40e030_dir_grid.zip', 'n40e035_dir_grid.zip', 'n40e040_dir_grid.zip', 'n40e045_dir_grid.zip', 'n40e050_dir_grid.zip', 'n40e055_dir_grid.zip', 'n40e060_dir_grid.zip', 'n40e065_dir_grid.zip', 'n40w005_dir_grid.zip', 'n40w010_dir_grid.zip', 'n45e000_dir_grid.zip', 'n45e005_dir_grid.zip', 'n45e010_dir_grid.zip', 'n45e015_dir_grid.zip', 'n45e020_dir_grid.zip', 'n45e025_dir_grid.zip', 'n45e030_dir_grid.zip', 'n45e035_dir_grid.zip', 'n45e040_dir_grid.zip', 'n45e045_dir_grid.zip', 'n45e050_dir_grid.zip', 'n45e055_dir_grid.zip', 'n45e060_dir_grid.zip', 'n45e065_dir_grid.zip', 'n45w005_dir_grid.zip', 'n45w010_dir_grid.zip', 'n50e000_dir_grid.zip', 'n50e005_dir_grid.zip', 'n50e010_dir_grid.zip', 'n50e015_dir_grid.zip', 'n50e020_dir_grid.zip', 'n50e025_dir_grid.zip', 'n50e030_dir_grid.zip', 'n50e035_dir_grid.zip', 'n50e040_dir_grid.zip', 'n50e045_dir_grid.zip', 'n50e050_dir_grid.zip', 'n50e055_dir_grid.zip', 'n50e060_dir_grid.zip', 'n50e065_dir_grid.zip', 'n50w005_dir_grid.zip', 'n50w010_dir_grid.zip', 'n50w015_dir_grid.zip', 'n55e000_dir_grid.zip', 'n55e005_dir_grid.zip', 'n55e010_dir_grid.zip', 'n55e015_dir_grid.zip', 'n55e020_dir_grid.zip', 'n55e025_dir_grid.zip', 'n55e030_dir_grid.zip', 'n55e035_dir_grid.zip', 'n55e040_dir_grid.zip', 'n55e045_dir_grid.zip', 'n55e050_dir_grid.zip', 'n55e055_dir_grid.zip', 'n55e060_dir_grid.zip', 'n55e065_dir_grid.zip', 'n55w005_dir_grid.zip', 'n55w010_dir_grid.zip', 'n55w015_dir_grid.zip']],
            'United States, Canada': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/NA/', ['n20w075_dir_grid.zip', 'n20w080_dir_grid.zip', 'n20w085_dir_grid.zip', 'n20w090_dir_grid.zip', 'n20w095_dir_grid.zip', 'n20w100_dir_grid.zip', 'n20w105_dir_grid.zip', 'n20w110_dir_grid.zip', 'n20w115_dir_grid.zip', 'n20w120_dir_grid.zip', 'n25w080_dir_grid.zip', 'n25w085_dir_grid.zip', 'n25w090_dir_grid.zip', 'n25w095_dir_grid.zip', 'n25w100_dir_grid.zip', 'n25w105_dir_grid.zip', 'n25w110_dir_grid.zip', 'n25w115_dir_grid.zip', 'n25w120_dir_grid.zip', 'n30w080_dir_grid.zip', 'n30w085_dir_grid.zip', 'n30w090_dir_grid.zip', 'n30w095_dir_grid.zip', 'n30w100_dir_grid.zip', 'n30w105_dir_grid.zip', 'n30w110_dir_grid.zip', 'n30w115_dir_grid.zip', 'n30w120_dir_grid.zip', 'n30w125_dir_grid.zip', 'n35w075_dir_grid.zip', 'n35w080_dir_grid.zip', 'n35w085_dir_grid.zip', 'n35w090_dir_grid.zip', 'n35w095_dir_grid.zip', 'n35w100_dir_grid.zip', 'n35w105_dir_grid.zip', 'n35w110_dir_grid.zip', 'n35w115_dir_grid.zip', 'n35w120_dir_grid.zip', 'n35w125_dir_grid.zip', 'n40w060_dir_grid.zip', 'n40w065_dir_grid.zip', 'n40w070_dir_grid.zip', 'n40w075_dir_grid.zip', 'n40w080_dir_grid.zip', 'n40w085_dir_grid.zip', 'n40w090_dir_grid.zip', 'n40w095_dir_grid.zip', 'n40w100_dir_grid.zip', 'n40w105_dir_grid.zip', 'n40w110_dir_grid.zip', 'n40w115_dir_grid.zip', 'n40w120_dir_grid.zip', 'n40w125_dir_grid.zip', 'n45w055_dir_grid.zip', 'n45w060_dir_grid.zip', 'n45w065_dir_grid.zip', 'n45w070_dir_grid.zip', 'n45w075_dir_grid.zip', 'n45w080_dir_grid.zip', 'n45w085_dir_grid.zip', 'n45w090_dir_grid.zip', 'n45w095_dir_grid.zip', 'n45w100_dir_grid.zip', 'n45w105_dir_grid.zip', 'n45w110_dir_grid.zip', 'n45w115_dir_grid.zip', 'n45w120_dir_grid.zip', 'n45w125_dir_grid.zip', 'n45w130_dir_grid.zip', 'n50w060_dir_grid.zip', 'n50w065_dir_grid.zip', 'n50w070_dir_grid.zip', 'n50w075_dir_grid.zip', 'n50w080_dir_grid.zip', 'n50w085_dir_grid.zip', 'n50w090_dir_grid.zip', 'n50w095_dir_grid.zip', 'n50w100_dir_grid.zip', 'n50w105_dir_grid.zip', 'n50w110_dir_grid.zip', 'n50w115_dir_grid.zip', 'n50w120_dir_grid.zip', 'n50w125_dir_grid.zip', 'n50w130_dir_grid.zip', 'n50w135_dir_grid.zip', 'n55w060_dir_grid.zip', 'n55w065_dir_grid.zip', 'n55w070_dir_grid.zip', 'n55w075_dir_grid.zip', 'n55w080_dir_grid.zip', 'n55w085_dir_grid.zip', 'n55w090_dir_grid.zip', 'n55w095_dir_grid.zip', 'n55w100_dir_grid.zip', 'n55w105_dir_grid.zip', 'n55w110_dir_grid.zip', 'n55w115_dir_grid.zip', 'n55w120_dir_grid.zip', 'n55w125_dir_grid.zip', 'n55w130_dir_grid.zip', 'n55w135_dir_grid.zip', 'n55w140_dir_grid.zip', 'n55w145_dir_grid.zip']],
            'South America': ['http://earlywarning.usgs.gov/hydrodata/sa_dir_3s_zip_grid/SA/', ['n00w050_dir_grid.zip', 'n00w055_dir_grid.zip', 'n00w060_dir_grid.zip', 'n00w065_dir_grid.zip', 'n00w070_dir_grid.zip', 'n00w075_dir_grid.zip', 'n00w080_dir_grid.zip', 'n00w085_dir_grid.zip', 'n00w090_dir_grid.zip', 'n00w095_dir_grid.zip', 'n05w055_dir_grid.zip', 'n05w060_dir_grid.zip', 'n05w065_dir_grid.zip', 'n05w070_dir_grid.zip', 'n05w075_dir_grid.zip', 'n05w080_dir_grid.zip', 'n05w085_dir_grid.zip', 'n05w090_dir_grid.zip', 'n10w060_dir_grid.zip', 'n10w065_dir_grid.zip', 'n10w070_dir_grid.zip', 'n10w075_dir_grid.zip', 'n10w080_dir_grid.zip', 'n10w085_dir_grid.zip', 'n10w090_dir_grid.zip', 'n10w095_dir_grid.zip', 'n10w110_dir_grid.zip', 's05w035_dir_grid.zip', 's05w040_dir_grid.zip', 's05w045_dir_grid.zip', 's05w050_dir_grid.zip', 's05w055_dir_grid.zip', 's05w060_dir_grid.zip', 's05w065_dir_grid.zip', 's05w070_dir_grid.zip', 's05w075_dir_grid.zip', 's05w080_dir_grid.zip', 's05w085_dir_grid.zip', 's05w090_dir_grid.zip', 's05w095_dir_grid.zip', 's10w035_dir_grid.zip', 's10w040_dir_grid.zip', 's10w045_dir_grid.zip', 's10w050_dir_grid.zip', 's10w055_dir_grid.zip', 's10w060_dir_grid.zip', 's10w065_dir_grid.zip', 's10w070_dir_grid.zip', 's10w075_dir_grid.zip', 's10w080_dir_grid.zip', 's10w085_dir_grid.zip', 's15w040_dir_grid.zip', 's15w045_dir_grid.zip', 's15w050_dir_grid.zip', 's15w055_dir_grid.zip', 's15w060_dir_grid.zip', 's15w065_dir_grid.zip', 's15w070_dir_grid.zip', 's15w075_dir_grid.zip', 's15w080_dir_grid.zip', 's20w040_dir_grid.zip', 's20w045_dir_grid.zip', 's20w050_dir_grid.zip', 's20w055_dir_grid.zip', 's20w060_dir_grid.zip', 's20w065_dir_grid.zip', 's20w070_dir_grid.zip', 's20w075_dir_grid.zip', 's20w080_dir_grid.zip', 's25w045_dir_grid.zip', 's25w050_dir_grid.zip', 's25w055_dir_grid.zip', 's25w060_dir_grid.zip', 's25w065_dir_grid.zip', 's25w070_dir_grid.zip', 's25w075_dir_grid.zip', 's30w050_dir_grid.zip', 's30w055_dir_grid.zip', 's30w060_dir_grid.zip', 's30w065_dir_grid.zip', 's30w070_dir_grid.zip', 's30w075_dir_grid.zip', 's30w080_dir_grid.zip', 's30w085_dir_grid.zip', 's35w055_dir_grid.zip', 's35w060_dir_grid.zip', 's35w065_dir_grid.zip', 's35w070_dir_grid.zip', 's35w075_dir_grid.zip', 's35w080_dir_grid.zip', 's35w085_dir_grid.zip', 's40w060_dir_grid.zip', 's40w065_dir_grid.zip', 's40w070_dir_grid.zip', 's40w075_dir_grid.zip', 's45w065_dir_grid.zip', 's45w070_dir_grid.zip', 's45w075_dir_grid.zip', 's45w080_dir_grid.zip', 's50w070_dir_grid.zip', 's50w075_dir_grid.zip', 's50w080_dir_grid.zip', 's55w060_dir_grid.zip', 's55w065_dir_grid.zip', 's55w070_dir_grid.zip', 's55w075_dir_grid.zip', 's55w080_dir_grid.zip', 's60w070_dir_grid.zip', 's60w075_dir_grid.zip']]
        }
        
        urls, tiles, lats, lons = [], [], [], []
        for continent in dire:
            for tile in dire[continent][1]:
                lat = int(tile[1:3])
                if tile[0] == 's':
                    lat = -lat
                lon = int(tile[4:7])
                if tile[3] == 'w':
                    lon = -lon
                if tile not in tiles:
                    lats.append(lat + 5) # upper left
                    lons.append(lon)
                    tiles.append(tile)
                    urls.append(dire[continent][0])
        df = DataFrame({'lat': lats, 'lon': lons, 'tile': tiles, 'url': urls}).sort_values(by=['lat', 'lon'], ascending = [0, 1])  # top-down, left-right
        df['done1'] = False
        df['done2'] = True
        df.index = range(len(df))
        df.to_pickle('tmp/df.pkl')

    while (not np.all(df.done1)) or (not np.all(df.done2)):
        if not np.all(df.done1):
            row = df[df.done1==False].iloc[0]
        else:
            if not os.path.exists('tiles/acc_pass1'):
                shutil.copytree('tiles/acc', 'tiles/acc_pass1')
            row = df[df.done2==False].iloc[0]
        lat, lon = row['lat'], row['lon']
        flow_dir = get_flow_dir(row, dir_path)
        name = row['tile'][:-len('_dir_grid.zip')]
        try:
            flow_acc = np.load(acc_path + name + '_acc.npz')['a']
        except:
            flow_acc = np.zeros((6000, 6000), dtype='uint32')
        try:
            udlr_in = np.load(f'{urld_path}/udlr_{lat}_{lon}.npz')['a']
        except:
            udlr_in = np.zeros((4, 6000), dtype='uint32')
        udlr_out = np.zeros((4, 6000+2), dtype='uint32')
        do_inside = not df.loc[(df.lat==lat) & (df.lon==lon), 'done1'].values[0]
        print(f'Processing {name} (inside: {do_inside})')
        for row_i in tqdm(range(6000)):
            drop_pixel(flow_dir, flow_acc, udlr_in, udlr_out, do_inside, row_i)
        np.savez_compressed(acc_path + name + '_acc', a=flow_acc)
        try:
            os.remove(f'{udlr_path}/udlr_{lat}_{lon}.npz')
        except:
            pass
        df.loc[(df.lat==lat) & (df.lon==lon), 'done1'] = True
        df.loc[(df.lat==lat) & (df.lon==lon), 'done2'] = True
        var = [[5, 0, 1, 0, (0, 0), (1, -1), 5, -5], [-5, 0, 0, 1, (0, -1), (1, 0), 5, 5], [0, -5, 3, 2, (1, 0), (0, -1), -5, -5], [0, 5, 2, 3, (1, -1), (0, 0), -5, 5]]
        for i in range(4):
            if np.any(udlr_out[i][1:-1]):
                lat2 = lat + var[i][0]
                lon2 = lon + var[i][1]
                df.loc[(df.lat==lat2) & (df.lon==lon2), 'done2'] = False
                udlr_name = f'{udlr_path}/udlr_{lat2}_{lon2}'
                try:
                    udlr = np.load(f'{udlr_name}.npz')['a']
                except:
                    udlr = np.zeros((4, 6000), dtype='uint32')
                udlr[var[i][2]] += udlr_out[var[i][3]][1:-1]
                np.savez_compressed(udlr_name, a=udlr)
            if udlr_out[var[i][4]] != 0:
                lat2 = lat + var[i][6]
                lon2 = lon + var[i][7]
                df.loc[(df.lat==lat2) & (df.lon==lon2), 'done2'] = False
                udlr_name = f'{udlr_path}/udlr_{lat2}_{lon2}'
                try:
                    udlr = np.load(f'{udlr_name}.npz')['a']
                except:
                    udlr = np.zeros((4, 6000), dtype='uint32')
                udlr[var[i][5]] += udlr_out[var[i][4]]
                np.savez_compressed(udlr_name, a=udlr)
        df.to_pickle('tmp/df.pkl')

if __name__ == '__main__':
    acc_flow()
