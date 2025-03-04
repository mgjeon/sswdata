start_date="2015-12-19"
end_date="2024-01-01"
cadence="1days"
wavelengths="171,193,304"
ds_path="/mnt/f/data/sdo/aia"
python -m itipy.download.download_sdo_aia_euv --ds_path $ds_path --start_date $start_date --end_date $end_date --cadence $cadence --wavelengths $wavelengths