"""
@author: Mingyu Jeon (mgjeon@khu.ac.kr)

Adapted from the ITI code by Robert Jarolim
Reference:
1) https://github.com/RobertJaro/InstrumentToInstrument
"""

import yaml
import logging
import argparse
import multiprocessing
from pathlib import Path
from datetime import timedelta
from urllib import request
import warnings
warnings.filterwarnings("ignore")

import drms
import numpy as np
import pandas as pd
from astropy.io import fits
from sunpy.io._fits import header_to_fits
from sunpy.util import MetaDict
from sunpy.map import Map

from tqdm import tqdm
# https://github.com/tqdm/tqdm?tab=readme-ov-file#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize
    
def download_url(url, filename):
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=url.split('/')[-1]) as t:  # all optional kwargs
        request.urlretrieve(url, filename=filename,
                            reporthook=t.update_to, data=None)
        t.total = t.n

class SDODownloader:
    """
    Class to download SDO EUV data from JSOC.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
        email (str): Email address for JSOC registration.
        wavelengths (list): List of wavelengths to download.
        n_workers (int): Number of worker threads for parallel download.
    """
    def __init__(
        self, 
        base_path,
        email,
        wavelengths=['94', '131', '171', '193', '211', '304', '335', '1600', '1700', 'hmi'], 
        n_workers=5
    ):
        self.ds_path = Path(base_path)
        self.n_workers = n_workers
        self.drms_client = drms.Client(email=email)
        self.wavelengths = [str(wl) for wl in wavelengths]

        assert all([wl in ['94', '131', '171', '193', '211', '304', '335', '1600', '1700', 'hmi'] for wl in self.wavelengths]), \
            'Invalid wavelength(s). Only 94, 131, 171, 193, 211, 304, 335, 1600, 1700, hmi are supported.'
        
        self.euv_wavelengths = [str(wl) for wl in wavelengths if wl != 'hmi' and wl != '1600' and wl != '1700']
        self.uv_wavelengths = [str(wl) for wl in wavelengths if wl == '1600' or wl == '1700']
        self.hmi_wavelengths = ['hmi'] if 'hmi' in wavelengths else []

        [(self.ds_path / str(wl)).mkdir(parents=True, exist_ok=True) for wl in self.euv_wavelengths] if len(self.euv_wavelengths) > 0 else None
        [(self.ds_path / str(wl)).mkdir(parents=True, exist_ok=True) for wl in self.uv_wavelengths] if len(self.uv_wavelengths) > 0 else None
        [(self.ds_path / str(wl)).mkdir(parents=True, exist_ok=True) for wl in self.hmi_wavelengths] if len(self.hmi_wavelengths) > 0 else None

        self.setup_logging()
    
    def setup_logging(self):
        """
        Setup the logging for the downloader.
        """
        log_file = self.ds_path / 'log.txt'
        logger = logging.getLogger('SDODownloader')
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        self.logger = logger

    def download(self, sample):
        """
        Download the data from JSOC.

        Args:
            sample (tuple): Tuple containing the header, segment and time information.

        Returns:
            str: Path to the downloaded file.
        """
        header, segment, t = sample
        try:
            wavelnth = 'hmi' if header['WAVELNTH'] == 6173 else header['WAVELNTH']
            ds_dir = self.ds_path / str(wavelnth)
            map_path = ds_dir / '{}.fits'.format(t.isoformat('T', timespec='seconds').replace(':', '-'))
            if map_path.exists():
                self.logger.info('Already downloaded (%s) %s' % (wavelnth, header['DATE__OBS']))
                return map_path
            # load map
            url = 'http://jsoc.stanford.edu' + segment
            # request.urlretrieve(url, filename=map_path)
            download_url(url, filename=map_path)

            header['DATE_OBS'] = header['DATE__OBS']
            header = header_to_fits(MetaDict(header))
            with fits.open(map_path, 'update') as f:
                hdr = f[1].header
                for k, v in header.items():
                    if pd.isna(v):
                        continue
                    hdr[k] = v
                f.verify('silentfix')

            return map_path
        except Exception as ex:
            self.logger.info('Download failed: %s (requeue)' % header['DATE__OBS'])
            self.logger.info(ex)
            raise ex

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        flag = 0
        for wl in self.wavelengths:
            map_path = self.ds_path / str(wl) / '{}.fits'.format(date.isoformat('T', timespec='seconds'))
            if map_path.exists():
                try:
                    Map(map_path)
                except Exception as ex:
                    self.logger.info('Invalid file: %s %s' % (map_path, ex))
                    map_path.unlink()
                    continue
                flag += 1
        if flag == len(self.wavelengths):
            self.logger.info('Already downloaded: %s' % id)
        else:
            self.logger.info('Start download: %s' % id)

            time_param = '%sZ' % date.isoformat('_', timespec='seconds')

            queue = []

            if len(self.euv_wavelengths) > 0:
                # query EUV
                ds_euv = 'aia.lev1_euv_12s[%s][%s]{image}' % (time_param, ','.join(self.euv_wavelengths))
                keys_euv = self.drms_client.keys(ds_euv)
                header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
                if len(header_euv) != len(self.euv_wavelengths) or np.any(header_euv.QUALITY != 0):
                    self.fetchDataFallback(date, kind='euv')
                    header_euv, segment_euv = None, None
                if header_euv is not None and segment_euv is not None:
                    for (idx, h), s in zip(header_euv.iterrows(), segment_euv.image):
                        queue += [(h.to_dict(), s, date)]

            if len(self.uv_wavelengths) > 0:
                # query UV
                ds_uv = 'aia.lev1_uv_24s[%s][%s]{image}' % (time_param, ','.join(self.uv_wavelengths))
                keys_uv = self.drms_client.keys(ds_uv)
                header_uv, segment_uv = self.drms_client.query(ds_uv, key=','.join(keys_uv), seg='image')
                if len(header_uv) != len(self.uv_wavelengths) or np.any(header_uv.QUALITY != 0):
                    self.fetchDataFallback(date, kind='uv')
                    header_uv, segment_uv = None, None
                if header_uv is not None and segment_uv is not None:
                    for (idx, h), s in zip(header_uv.iterrows(), segment_uv.image):
                        queue += [(h.to_dict(), s, date)]

            if len(self.hmi_wavelengths) > 0:
                # query HMI
                ds_hmi = 'hmi.M_720s[%s]{magnetogram}' % time_param
                keys_hmi = self.drms_client.keys(ds_hmi)
                header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
                if len(header_hmi) != 1 or np.any(header_hmi.QUALITY != 0):
                    self.fetchDataFallback(date, kind='hmi')
                    header_hmi, segment_hmi = None, None
                if header_hmi is not None and segment_hmi is not None:
                    for (idx, h), s in zip(header_hmi.iterrows(), segment_hmi.magnetogram):
                        queue += [(h.to_dict(), s, date)]
                        
            if len(queue) == 0:
                return

            with multiprocessing.Pool(self.n_workers) as p:
                p.map(self.download, queue)

            self.logger.info('Finished: %s' % id)

    def sort_header_segment(self, date, header_tmp, segment_tmp):
        date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
        date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - date).abs()
        # sort and filter
        header_tmp['date_diff'] = date_diff
        header_tmp = header_tmp.sort_values('date_diff')
        segment_tmp['date_diff'] = date_diff
        segment_tmp = segment_tmp.sort_values('date_diff')
        cond_tmp = header_tmp.QUALITY == 0
        header_tmp = header_tmp[cond_tmp]
        segment_tmp = segment_tmp[cond_tmp]
        return header_tmp, segment_tmp

    def fetchDataFallback(self, date, kind=None):
        """
        Download the data for the given date using fallback.

        Args:
            date (datetime): The date for which the data should be downloaded.
            kind (str): The kind of data to download (euv, uv, hmi).

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        self.logger.info('Fallback download: %s' % id)

        if kind == 'euv':
            # query EUV
            header_euv, segment_euv = [], []
            t = date - timedelta(seconds=24)
            for wl in self.euv_wavelengths:
                map_path = self.ds_path / str(wl) / '{}.fits'.format(date.isoformat('T', timespec='seconds'))
                if map_path.exists():
                    try:
                        Map(map_path)
                        self.logger.info('Already downloaded: %s %s' % (wl, date))
                        continue
                    except Exception as ex:
                        self.logger.info('Invalid file: %s %s' % (map_path, ex))
                        map_path.unlink()
                self.logger.info('Find nearest time with 48s window: %s %s' % (wl, date))
                ds_euv = 'aia.lev1_euv_12s[%sZ/48s][%s]{image}' % (
                    t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
                keys_euv = self.drms_client.keys(ds_euv)
                header_tmp, segment_tmp = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
                # assert len(header_tmp) != 0, 'No data found!'
                if len(header_tmp) == 0:
                    self.logger.info('No data found: %s %s' % (wl, date))
                    continue
                header_tmp, segment_tmp = self.sort_header_segment(date, header_tmp, segment_tmp)
                # assert len(header_tmp) > 0, 'No valid quality flag found'
                if len(header_tmp) == 0:
                    self.logger.info('No valid quality flag found: %s %s' % (wl, date))
                    continue
                # replace invalid
                header_euv.append(header_tmp.iloc[0].drop('date_diff'))
                segment_euv.append(segment_tmp.iloc[0].drop('date_diff'))
                self.logger.info('Found time: %s %s' % (wl, header_tmp.iloc[0]['DATE__OBS']))

            if len(header_euv) != len(self.wavelengths):
                self.logger.info('No valid data found: %s' % id)

            queue = []
            for h, s in zip(header_euv, segment_euv):
                queue += [(h.to_dict(), s.image, date)]

            with multiprocessing.Pool(self.n_workers) as p:
                p.map(self.download, queue)

            self.logger.info('Finished: %s' % id)

        elif kind == 'uv':
            # query UV
            header_uv, segment_uv = [], []
            t = date - timedelta(seconds=48)
            for wl in self.uv_wavelengths:
                map_path = self.ds_path / str(wl) / '{}.fits'.format(date.isoformat('T', timespec='seconds'))
                if map_path.exists():
                    try:
                        Map(map_path)
                        self.logger.info('Already downloaded: %s %s' % (wl, date))
                        continue
                    except Exception as ex:
                        self.logger.info('Invalid file: %s %s' % (map_path, ex))
                        map_path.unlink()
                self.logger.info('Find nearest time with 96s window: %s %s' % (wl, date))
                ds_uv = 'aia.lev1_uv_24s[%sZ/96s][%s]{image}' % (
                    t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
                keys_uv = self.drms_client.keys(ds_uv)
                header_tmp, segment_tmp = self.drms_client.query(ds_uv, key=','.join(keys_uv), seg='image')
                # assert len(header_tmp) != 0, 'No data found!'
                if len(header_tmp) == 0:
                    self.logger.info('No data found: %s %s' % (wl, date))
                    continue
                header_tmp, segment_tmp = self.sort_header_segment(date, header_tmp, segment_tmp)
                # assert len(header_tmp) > 0, 'No valid quality flag found'
                if len(header_tmp) == 0:
                    self.logger.info('No valid quality flag found: %s %s' % (wl, date))
                    continue
                # replace invalid
                header_uv.append(header_tmp.iloc[0].drop('date_diff'))
                segment_uv.append(segment_tmp.iloc[0].drop('date_diff'))
                self.logger.info('Found time: %s %s' % (wl, header_tmp.iloc[0]['DATE__OBS']))

            if len(header_uv) != len(self.uv_wavelengths):
                self.logger.info(f'No valid data found: {id} (UV)')

            queue = []
            for h, s in zip(header_uv, segment_uv):
                queue += [(h.to_dict(), s.image, date)]
            
            with multiprocessing.Pool(self.n_workers) as p:
                p.map(self.download, queue)

            self.logger.info('Finished: %s' % id)

        elif kind == 'hmi':
            # query HMI
            t = date - timedelta(minutes=24)
            map_path = self.ds_path / 'hmi' / '{}.fits'.format(date.isoformat('T', timespec='seconds'))
            if map_path.exists():
                try:
                    Map(map_path)
                    self.logger.info(f'Already downloaded: hmi {date}')
                    return
                except Exception as ex:
                    self.logger.info('Invalid file: %s %s' % (map_path, ex))
                    map_path.unlink()
            self.logger.info(f'Find nearest time with 48m window: hmi {date}')
            ds_hmi = 'hmi.M_720s[%sZ/48m]{magnetogram}' % t.replace(tzinfo=None).isoformat('_', timespec='seconds')
            keys_hmi = self.drms_client.keys(ds_hmi)
            header_tmp, segment_tmp = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg='magnetogram')
            # assert len(header_tmp) != 0, 'No data found!'
            if len(header_tmp) == 0:
                self.logger.info(f'No data found: hmi {date}')
                return
            header_tmp, segment_tmp = self.sort_header_segment(date, header_tmp, segment_tmp)
            # assert len(header_tmp) > 0, 'No valid quality flag found'
            if len(header_tmp) == 0:
                self.logger.info(f'No valid quality flag found: hmi {date}')
                return
            # replace invalid
            header_hmi = header_tmp.iloc[0].drop('date_diff')
            segment_hmi = segment_tmp.iloc[0].drop('date_diff')
            self.logger.info('Found time: hmi %s' % (header_tmp.iloc[0]['DATE__OBS']))

            queue = [(header_hmi.to_dict(), segment_hmi.magnetogram, date)]

            with multiprocessing.Pool(self.n_workers) as p:
                p.map(self.download, queue)

            self.logger.info('Finished: %s' % id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/download_sdo.yaml")

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    download_dir = config["download_dir"]
    email = config["email"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    wavelengths = config["wavelengths"]
    n_workers = config["n_workers"]

    downloader = SDODownloader(base_path=download_dir, 
                               email=email,
                               wavelengths=wavelengths,
                               n_workers=n_workers)

    interval = config["interval"]

    value = interval[0]
    unit = interval[1]

    if unit == "days":
        dt = timedelta(days=value)
    elif unit == "hours":
        dt = timedelta(hours=value)
    elif unit == "minutes":
        dt = timedelta(minutes=value)
    else:
        raise ValueError("Invalid unit. Only days, hours, and minutes are supported.")

    for d in [start_date + i * dt for i in
              range((end_date - start_date) // dt)]:
        downloader.downloadDate(d)
