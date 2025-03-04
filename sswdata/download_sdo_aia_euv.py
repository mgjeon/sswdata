import logging
import argparse
import multiprocessing
from pathlib import Path
from datetime import timedelta, datetime

import drms
import numpy as np
import pandas as pd

from itipy.download.base_sdo_jsoc import BaseSDOJSOCDownloader


class SDOAIAEUVDownloader(BaseSDOJSOCDownloader):
    """
    Class to download SDO/AIA EUV data from JSOC.
    
    Products:
        aia.lev1_euv_12s
        AIA Level 1, 12 second cadence EUV images
        94 Å, 131 Å, 171 Å, 193 Å, 211 Å, 304 Å, 335 Å

    Args:
        ds_path (str): Path to the directory where the downloaded data should be stored.
        n_workers (int): Number of worker threads for parallel download.
        wavelengths (list): List of wavelengths to download.
        headertime (book): Whether to use the header time for the file name.
        fallback (bool): Whether to use fallback download if quality check fails.
    """
    def __init__(self, ds_path, n_workers=1,
                 wavelengths=[94, 131, 171, 193, 211, 304, 335],
                 headertime=False,
                 fallback=False):
        self.ds_path = ds_path
        self.n_workers = n_workers
        self.headertime = headertime
        self.fallback = fallback

        self.wavelengths = [str(wl) for wl in wavelengths]
        [(Path(ds_path) / wl).mkdir(parents=True, exist_ok=True) for wl in self.wavelengths]

        self.drms_client = drms.Client()

        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True, 
                            handlers=[logging.FileHandler(f"{ds_path}/info.log"), logging.StreamHandler()])
        self.logger = logging.getLogger('SDOAIAEUVDownloader')

    def set_dir_desc(self):
        header, segment, t = self.sample
        dir = Path(self.ds_path) / str(header['WAVELNTH'])
        desc = str(header['WAVELNTH'])
        return dir, desc

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()
        self.logger.info('Start download: %s' % id)

        time_param = '%sZ' % date.isoformat('_', timespec='seconds')
        ds_euv = 'aia.lev1_euv_12s[%s][%s]{image}' % (time_param, ','.join(self.wavelengths))
        keys_euv = self.drms_client.keys(ds_euv)
        header_euv, segment_euv = self.drms_client.query(ds_euv, key=','.join(keys_euv), seg='image')
        if len(header_euv) != len(self.wavelengths) or np.any(header_euv.QUALITY != 0):
            if self.fallback:
                self.fetchDataFallback(date)
                return
            else:
                self.logger.info("Skipping: %s" % id)
                return

        queue = []
        for (idx, h), seg in zip(header_euv.iterrows(), segment_euv.image):
            queue += [(h.to_dict(), seg, date)]

        try:
            with multiprocessing.Pool(self.n_workers) as p:
                p.map(self.download, queue)
        except Exception as ex:
            self.logger.error(ex)
            for q in queue:
                try:
                    self.download(q)
                except Exception as ex:
                    self.logger.error(ex)
                    self.logger.error('Failed: %s' % id)
                    continue
        self.logger.info('Finished: %s' % id)

    def fetchDataFallback(self, date):
        """
        Download the data for the given date using fallback.

        Args:
            date (datetime): The date for which the data should be downloaded.

        Returns:
            list: List of paths to the downloaded files.
        """
        id = date.isoformat()

        self.logger.info('Fallback download: %s' % id)
        # query EUV
        header_euv, segment_euv = [], []
        t = date - timedelta(hours=6)
        for wl in self.wavelengths:
            euv_ds = 'aia.lev1_euv_12s[%sZ/12h@12s][%s]{image}' % (
                t.replace(tzinfo=None).isoformat('_', timespec='seconds'), wl)
            keys_euv = self.drms_client.keys(euv_ds)
            header_tmp, segment_tmp = self.drms_client.query(euv_ds, key=','.join(keys_euv), seg='image')
            assert len(header_tmp) != 0, 'No data found!'
            date_str = header_tmp['DATE__OBS'].replace('MISSING', '').str.replace('60', '59')  # fix date format
            date_diff = (pd.to_datetime(date_str).dt.tz_localize(None) - date).abs()
            # sort and filter
            header_tmp['date_diff'] = date_diff
            header_tmp.sort_values('date_diff')
            segment_tmp['date_diff'] = date_diff
            segment_tmp.sort_values('date_diff')
            cond_tmp = header_tmp.QUALITY == 0
            header_tmp = header_tmp[cond_tmp]
            segment_tmp = segment_tmp[cond_tmp]
            assert len(header_tmp) > 0, 'No valid quality flag found'
            # replace invalid
            header_euv.append(header_tmp.iloc[0].drop('date_diff'))
            segment_euv.append(segment_tmp.iloc[0].drop('date_diff'))

        queue = []
        for h, s in zip(header_euv, segment_euv):
            queue += [(h.to_dict(), s.image, date)]

        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)

        self.logger.info('Finished: %s' % id)


if __name__ == '__main__':
    from itipy.download.util import get_timedelta

    parser = argparse.ArgumentParser(description='Download SDO/AIA EUV data from JSOC with quality check and fallback')
    parser.add_argument('--ds_path', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=7)
    parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False, default=str(datetime.now()).split(' ')[0])
    parser.add_argument('--cadence', type=str, help='cadence for the download.', required=False, default="1days")
    parser.add_argument('--wavelengths', type=str, help='wavelengths to download.', required=False, default="94,131,171,193,211,304,335")
    parser.add_argument('--headertime', type=bool, help='use header time for file name.', required=False, default=False)
    parser.add_argument('--fallback', type=bool, help='use fallback download.', required=False, default=False)
    wavelengths = [int(wl) for wl in parser.parse_args().wavelengths.split(',')]
    print(wavelengths)

    args = parser.parse_args()

    downloader = SDOAIAEUVDownloader(ds_path=args.ds_path, n_workers=args.n_workers, wavelengths=wavelengths, headertime=args.headertime, fallback=args.fallback)

    t_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    t_end = datetime.strptime(args.end_date, "%Y-%m-%d")
    td = get_timedelta(args.cadence)
    date_list = [t_start + i * td for i in range((t_end - t_start) // td)]

    import warnings; warnings.filterwarnings("ignore")
    for d in date_list:
        downloader.downloadDate(d)
