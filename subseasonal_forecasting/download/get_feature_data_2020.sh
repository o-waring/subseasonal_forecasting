# height 2020
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.2020.nc -o hgt.2020.nc ; cd -; }

# # icec 2020 #WRONG - need PCA data
# cd data/prediction_inputs_raw/ && { curl -O ftp://ftp.cdc.noaa.gov/Projects/Datasets/noaa.oisst.v2.highres/icec.day.mean.2020.nc -o icec.day.mean.2020.nc ; cd -; }
# Perhaps - ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/

#
# # sst 2020 #WRONG - need PCA data
# cd data/prediction_inputs_raw/ && { curl -O ftp://ftp.cdc.noaa.gov/Projects/Datasets/noaa.oisst.v2.highres/sst.day.mean.2020.nc -o sst.day.mean.2020.nc ; cd -; }

# rhum at surface 2019
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/rhum.sig995.2020.nc -o rhum.sig995.2020.nc ; cd -; }

# pevpr 2020
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pevpr.sfc.gauss.2020.nc -o pevpr.sfc.gauss.2020.nc ; cd -; }

# pr_wtr 2020 precipitable water
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/pr_wtr.eatm.2020.nc -o pr_wtr.eatm.2020.nc ; cd -; }

# pres gauss 2020
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pres.sfc.gauss.2020.nc -o pres.sfc.gauss.2020.nc ; cd -; }

# sea level pressure 2020
cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/slp.2020.nc -o slp.2020.nc ; cd -; }

# MEI
#cd data/prediction_inputs_raw/ && { curl -O https://psl.noaa.gov/enso/mei/data/meiv2.data -o meiv2.data ; cd -; }
