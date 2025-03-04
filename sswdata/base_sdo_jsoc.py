import logging
from datetime import datetime

import pandas as pd
from astropy.io import fits
from sunpy.io._fits import header_to_fits
from sunpy.util import MetaDict

from itipy.download.util import download_url


class BaseSDOJSOCDownloader:
    """
    Base Class to download SDO data from JSOC.
    """

    def __init__(self):
        pass
    
    def download(self, sample):
        """
        Download the data from JSOC.

        Args:
            sample (tuple): Tuple containing the header, segment and time information.

        Returns:
            str: Path to the downloaded file.
        """
        self.sample = sample    
        header, segment, t = sample
        dir, desc = self.set_dir_desc()
        try:
            if self.headertime:
                tt = datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y%m%d_%H%M%S")
            else:
                tt = t.isoformat('T', timespec='seconds').replace(':', '')
            map_path = dir / ('%s.fits' % tt)
            if map_path.exists():
                return map_path
            # load map
            url = 'http://jsoc.stanford.edu' + segment
            download_url(url, filename=map_path, desc=desc)

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
        
