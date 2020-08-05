# height 2019
FILE=data/prediction_inputs/hgt.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.2019.nc -o hgt.2019.nc ; cd -; }
fi

# rhum at surface 2019
FILE=data/prediction_inputs/rhum.sig995.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/rhum.sig995.2019.nc -o rhum.sig995.2019.nc ; cd -; }
fi

# pevpr 2019
FILE=data/prediction_inputs/pevpr.sfc.gauss.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pevpr.sfc.gauss.2019.nc -o pevpr.sfc.gauss.2019.nc ; cd -; }
fi

# pr_wtr 2020 precipitable water
FILE=data/prediction_inputs/pr_wtr.eatm.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/pr_wtr.eatm.2019.nc -o pr_wtr.eatm.2019.nc ; cd -; }
fi

# pres gauss 2019
FILE=data/prediction_inputs/pres.sfc.gauss.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/pres.sfc.gauss.2019.nc -o pres.sfc.gauss.2019.nc ; cd -; }
fi

# Elevation
FILE=data/prediction_inputs/elev.1-deg.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O http://research.jisao.washington.edu/data_sets/elevation/elev.1-deg.nc -o elev.1-deg.nc ; cd -; }
fi

# sea level pressure 2019
FILE=data/prediction_inputs/slp.2019.nc
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
  cd data/prediction_inputs/ && { curl -O ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/surface/slp.2019.nc -o slp.2019.nc ; cd -; }
fi
