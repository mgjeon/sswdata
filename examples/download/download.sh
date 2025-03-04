#!/bin/bash

email="mgjeon@khu.ac.kr"
start_date="2024-01-01"
end_date="2024-01-02"
cadence="12hours"

# sdo_aia_euv
ds_path="./dataset/sdo/aia"
python -m itipy.download.download_sdo_aia_euv --ds_path $ds_path --email $email --start_date $start_date --end_date $end_date --cadence $cadence

# sdo_hmi
ds_path="./dataset/sdo/hmi"
python -m itipy.download.download_sdo_hmi --ds_path $ds_path --email $email --start_date $start_date --end_date $end_date --cadence $cadence

# stereo_euvi
ds_path="./dataset/stereo/euvi"
start_date="2011-01-01"
end_date="2011-01-02"
python -m itipy.download.download_stereo_secchi_euvi --ds_path $ds_path --start_date $start_date --end_date $end_date --cadence $cadence

# stereo_euvi
ds_path="./dataset/soho/eit"
start_date="2007-12-13"
end_date="2007-12-14"
python -m itipy.download.download_soho_eit --ds_path $ds_path --start_date $start_date --end_date $end_date --cadence $cadence