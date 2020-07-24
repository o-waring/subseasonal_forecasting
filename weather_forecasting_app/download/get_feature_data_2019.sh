# height 2019
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.2019.nc -o hgt.2019.nc ; cd -; }

# # icec 2019 #WRONG - need PCA data
# cd ../../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Projects/Datasets/noaa.oisst.v2.highres/icec.day.mean.2019.nc -o icec.day.mean.2019.nc ; cd -; }
#
# # sst 2019 #WRONG - need PCA data
# cd ../../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Projects/Datasets/noaa.oisst.v2.highres/sst.day.mean.2019.nc -o sst.day.mean.2019.nc ; cd -; }

# rhum at surface 2019
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/rhum.sig995.2019.nc -o rhum.sig995.2019.nc ; cd -; }

# pevpr 2019
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pevpr.sfc.gauss.2019.nc -o pevpr.sfc.gauss.2019.nc ; cd -; }

# pr_wtr 2020 precipitable water
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/pr_wtr.eatm.2019.nc -o pr_wtr.eatm.2019.nc ; cd -; }

# pres gauss 2019
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pres.sfc.gauss.2019.nc -o pres.sfc.gauss.2019.nc ; cd -; }

# Elevation
cd ../data/prediction_inputs/ && { curl -O http://research.jisao.washington.edu/data_sets/elevation/elev.1-deg.nc -o elev.1-deg.nc ; cd -; }

# sea level pressure 2019
cd ../data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/slp.2019.nc -o slp.2019.nc ; cd -; }
