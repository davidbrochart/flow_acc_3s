import os
import sys
import zipfile
import requests
import shutil
from osgeo import gdal
from pandas import DataFrame
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import click
from threading import Thread
import json

def get_flow_dir(row):
    if not os.path.exists(f'tiles/dir/3s/{row.tile}'):
        tqdm.write(f'Downloading {row.tile}...')
        r = requests.get(row.url + row.tile)
        with open(f'tiles/dir/3s/{row.tile}', 'wb') as f:
            f.write(r.content)
    try:
        with zipfile.ZipFile(f'tiles/dir/3s/{row.tile}', 'r') as z:
            z.extractall(path = 'tmp/')
        flow_dir = gdal.Open(f'tmp/{row.tile[:-9]}/{row.tile[:-9]}/w001001.adf')
        geo = flow_dir.GetGeoTransform()
        ySize, xSize = flow_dir.RasterYSize, flow_dir.RasterXSize
        flow_dir = flow_dir.ReadAsArray()
        shutil.rmtree(f'tmp/{row.tile[:-9]}')
        # data is padded into a 6000x6000 array (some tiles may be smaller):
        array_5x5 = np.zeros((6000, 6000), dtype = 'uint8')
        y0 = int(round((geo[3] - row.lat) / geo[5]))
        y1 = 6000 - int(round(((row.lat - 5) - (geo[3] + geo[5] * ySize)) / geo[5]))
        x0 = int(round((geo[0] - row.lon) / geo[1]))
        x1 = 6000 - int(round(((row.lon + 5) - (geo[0] + geo[1] * xSize)) / geo[1]))
        array_5x5[y0:y1, x0:x1] = flow_dir
    except:
        tqdm.write('Not a ZIP file!')
        array_5x5 = np.zeros((6000, 6000), dtype = 'uint8')
    return array_5x5

def pass1(cpu, drop_pixel, df):
    for row in df.iterrows():
        process_tile(cpu, drop_pixel, row[1], df, True)

def pass2(drop_pixel, df):
    while not np.all(df.done):
        row = df[df.done==False].iloc[0]
        process_tile(0, drop_pixel, row, df, False)

def process_tile(cpu, drop_pixel, row, df, first_pass):
    lat, lon = row['lat'], row['lon']
    flow_dir = get_flow_dir(row)
    name = row['tile'][:-len('_dir_grid.zip')]
    if os.path.exists(f'tiles/acc/3s/{name}_acc.npz'):
        flow_acc = np.load(f'tiles/acc/3s/{name}_acc.npz')['a']
    else:
        flow_acc = np.zeros((6000, 6000), dtype='uint32')
    if first_pass:
        udlr_in = np.zeros((4, 6000), dtype='uint32')
    else:
        df.loc[(df.lat==lat) & (df.lon==lon), 'done'] = True
        if os.path.exists(f'tmp/udlr/udlr_{lat}_{lon}.npz'):
            udlr_in = np.load(f'tmp/udlr/udlr_{lat}_{lon}.npz')['a']
        else:
            udlr_in = np.zeros((4, 6000), dtype='uint32')
    udlr_out = np.zeros((4, 6000+2), dtype='uint32')
    do_inside = first_pass
    tqdm.write(f'Processing {name} (inside: {do_inside})')
    row_nb = 60
    for row_i in tqdm(range(0, 6000, row_nb)):
        drop_pixel(flow_dir, flow_acc, udlr_in, udlr_out, do_inside, row_i, row_nb)
    np.savez_compressed(f'tiles/acc/3s/{name}_acc', a=flow_acc)
    if not first_pass:
        if os.path.exists(f'tmp/udlr/udlr_{lat}_{lon}.npz'):
            os.remove(f'tmp/udlr/udlr_{lat}_{lon}.npz')
    var = [[5, 0, 1, 0, (0, 0), (1, -1), 5, -5], [-5, 0, 0, 1, (0, -1), (1, 0), 5, 5], [0, -5, 3, 2, (1, 0), (0, -1), -5, -5], [0, 5, 2, 3, (1, -1), (0, 0), -5, 5]]
    for i in range(4):
        # do the sides
        if np.any(udlr_out[i][1:-1]):
            lat2 = lat + var[i][0]
            lon2 = lon + var[i][1]
            if not first_pass:
                df.loc[(df.lat==lat2) & (df.lon==lon2), 'done'] = False
            udlr_name = f'tmp/udlr{cpu}/udlr_{lat2}_{lon2}'
            if os.path.exists(f'{udlr_name}.npz'):
                udlr = np.load(f'{udlr_name}.npz')['a']
            else:
                udlr = np.zeros((4, 6000), dtype='uint32')
            udlr[var[i][2]] += udlr_out[var[i][3]][1:-1]
            np.savez_compressed(udlr_name, a=udlr)
        # do the corners
        if udlr_out[var[i][4]] != 0:
            lat2 = lat + var[i][6]
            lon2 = lon + var[i][7]
            if not first_pass:
                df.loc[(df.lat==lat2) & (df.lon==lon2), 'done'] = False
            udlr_name = f'tmp/udlr{cpu}/udlr_{lat2}_{lon2}'
            if os.path.exists(f'{udlr_name}.npz'):
                udlr = np.load(f'{udlr_name}.npz')['a']
            else:
                udlr = np.zeros((4, 6000), dtype='uint32')
            udlr[var[i][5]] += udlr_out[var[i][4]]
            np.savez_compressed(udlr_name, a=udlr)
    if first_pass:
        df.to_pickle(f'tmp/df{cpu}.pkl')
    else:
        df.to_pickle('tmp/df.pkl')

@click.command()
@click.option('-n', '--numba', is_flag=True, help='Use Numba as the computing backend (otherwise, use Cython).')
@click.option('-p', '--parallel', default=1, help='Number of CPU cores to use (for first pass only).')
@click.option('-r', '--reset', is_flag=1, help="Start the processing from scratch (don't download tiles if already downloaded).")
def acc_flow(numba, parallel, reset):
    if reset:
        shutil.rmtree('tmp', ignore_errors=True)
        shutil.rmtree('tiles/acc', ignore_errors=True)
    if numba:
        sys.path.append('numba')
        print('Using Numba')
    else:
        sys.path.append('cython')
        print('Using Cython')
    from drop_pixel import drop_pixel
    os.makedirs('tiles/dir/3s', exist_ok=True)
    os.makedirs('tiles/acc/3s', exist_ok=True)
    for cpu in range(parallel):
        os.makedirs(f'tmp/udlr{cpu}', exist_ok=True)

    if os.path.exists('tmp/df.pkl'):
        df = pd.read_pickle(f'tmp/df.pkl')
    else:
        # first pass
        df = []
        df_ok = True
        for cpu in range(parallel):
            if os.path.exists(f'tmp/df{cpu}.pkl'):
                df.append(pd.read_pickle(f'tmp/df{cpu}.pkl'))
            else:
                df_ok = False
        if not df_ok:
            with open('tiles.json') as f:
                dire = json.load(f)
            
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
            df.index = range(len(df))
            df_keep = df
            df = []
            i0 = 0
            size = len(df_keep) // parallel
            i1 = size
            for cpu in range(parallel):
                if cpu == parallel - 1:
                    i1 = None
                df.append(df_keep.iloc[i0:i1].copy(deep=True))
                df[cpu].to_pickle(f'tmp/df{cpu}.pkl')
                if cpu != parallel - 1:
                    i0 += size
                    i1 += size
        threads = []
        for cpu in range(parallel):
            t = Thread(target=pass1, args=(cpu, drop_pixel, df[cpu]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        df = pd.concat(df)
        df['done'] = False
        df.to_pickle('tmp/df.pkl')
        for cpu in range(parallel):
            for fname in os.listdir(f'tmp/udlr{cpu}'):
                udlr = np.load(f'tmp/udlr{cpu}/{fname}')['a']
                if os.path.exists(f'tmp/udlr/{fname}'):
                    udlr += np.load(f'tmp/udlr/{fname}')['a']
                np.savez_compressed(f'tmp/udlr/{fname[:-4]}', a=udlr)

    # second pass
    pass2(drop_pixel, df)

if __name__ == '__main__':
    acc_flow()
