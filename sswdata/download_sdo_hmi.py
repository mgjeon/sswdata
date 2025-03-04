import logging
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime

import drms
import numpy as np

from itipy.download.base_sdo_jsoc import BaseSDOJSOCDownloader


class SDOHMIDownloader(BaseSDOJSOCDownloader):
    """
    Class to download SDO/HMI Continuum & Magnetogram data from JSOC.

    Products:
        hmi.Ic_720s
        HMI Continuum Intensity 6173 Å, 720s cadence

        hmi.M_720s
        HMI Line-of-sight Magnetograms, 720s cadence

    Args:
        ds_path (str): Path to the directory where the downloaded data should be stored.
        n_workers (int): Number of worker threads for parallel download.
        ignore_quality (bool): If True, data with quality flag != 0 will be downloaded.
        series (str): Series name of the HMI LOS Magnetogram data.
    """
    def __init__(self, ds_path, n_workers=4, ignore_quality=False,
                 series=['Ic_720s', 'M_720s']):
        self.ds_path = ds_path
        self.ignore_quality = ignore_quality
        self.n_workers = n_workers

        self.series = series
        [(Path(ds_path) / s).mkdir(parents=True, exist_ok=True) for s in series]

        self.drms_client = drms.Client()

        logging.basicConfig(level=logging.INFO, 
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True, 
                            handlers=[logging.FileHandler(f"{ds_path}/info.log"), logging.StreamHandler()])
        self.logger = logging.getLogger('SDOHMIDownloader')

    def set_dir_desc(self):
        header, segment, t = self.sample
        content = header['CONTENT']
        cadence = str(int(header['CADENCE'])) + 's'
        if content == "CONTINUUM INTENSITY":
            prefix = 'Ic_'
        elif content == "MAGNETOGRAM":
            prefix = 'M_'
        else:
            raise Exception(f'content {content} not supported!')

        s = prefix + cadence
        dir = Path(self.ds_path) / s
        desc = s
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
        queue = []
        for s in self.series:
            if s == 'Ic_720s':
                segment = 'continuum'
            elif s == 'M_720s':
                segment = 'magnetogram'
            else:
                raise Exception(f'series hmi.{s} not supported!')

            ds_hmi = 'hmi.%s[%s]' % (s, time_param) + '{%s}' % segment
            keys_hmi = self.drms_client.keys(ds_hmi)
            header_hmi, segment_hmi = self.drms_client.query(ds_hmi, key=','.join(keys_hmi), seg=segment)
            if len(header_hmi) != 1 or (np.any(header_hmi.QUALITY != 0) and not self.ignore_quality):
                raise Exception('No valid data found!')

            if segment == 'continuum':
                queue += [(h.to_dict(), seg, date) for (idx, h), seg in zip(header_hmi.iterrows(), segment_hmi.continuum)]
            elif segment == 'magnetogram':
                queue += [(h.to_dict(), seg, date) for (idx, h), seg in zip(header_hmi.iterrows(), segment_hmi.magnetogram)]
            else:
                raise Exception(f'segment {segment} not supported!')
            
        with multiprocessing.Pool(self.n_workers) as p:
            p.map(self.download, queue)
        self.logger.info('Finished: %s' % id)


if __name__ == '__main__':
    from itipy.download.util import get_timedelta

    parser = argparse.ArgumentParser(description='Download SDO/HMI Continuum & Magnetogram data from JSOC with quality check')
    parser.add_argument('--ds_path', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=7)
    parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False, default=str(datetime.now()).split(' ')[0])
    parser.add_argument('--cadence', type=str, help='cadence for the download.', required=False, default="1days")
    
    args = parser.parse_args()

    downloader = SDOHMIDownloader(ds_path=args.ds_path, n_workers=args.n_workers)

    t_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    t_end = datetime.strptime(args.end_date, "%Y-%m-%d")
    td = get_timedelta(args.cadence)
    date_list = [t_start + i * td for i in range((t_end - t_start) // td)]

    import warnings; warnings.filterwarnings("ignore")
    for d in date_list:
        downloader.downloadDate(d)