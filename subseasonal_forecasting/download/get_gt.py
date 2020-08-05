import sys 
import os
import math 
import copy
from datetime import datetime
import struct

NA = -999
RESLAT = 360 
RESLON = 720
GS = RESLAT * RESLON
WEEKS2 = 14
WEEKS4 = 28
PI = 3.141592653589793238

def fileSize(name):
  statinfo = os.stat(name)
  return statinfo.st_size 

def folderCreate(path):
  try:
    os.mkdir(path)
    return False
  except OSError:
    return True

def splitBy(text, by):
  return text.split(by)

def weeks2sum(source, start):
  ans = 0
  for i in range(start, start + WEEKS2):
    if source[i] == NA:
      return NA
    ans += source[i]
  return ans

class PlaceData:
  lat = 0
  lon = 0
  prec34 = NA
  prec56 = NA
  temp34 = NA
  temp56 = NA
  tmax = []
  tmin = []
  prec = []
  tmaxNA = []
  tminNA = []
  precNA = []
  def __init__(self, line):
    row = splitBy(line, ',')
    self.lat = int(row[0])
    self.lon = int(row[1])
    self.tmax = [NA] * WEEKS4
    self.tmin = [NA] * WEEKS4
    self.prec = [NA] * WEEKS4
    self.tmaxNA = [NA] * WEEKS4
    self.tminNA = [NA] * WEEKS4
    self.precNA = [NA] * WEEKS4

  def interpolateDay(self, day, source):
    value = [0] * 4
    sourceLat = [self.lat - 0.25, self.lat - 0.25, self.lat + 0.25, self.lat + 0.25]
    value[0] = source[day * GS + ((self.lat + 90) * 2 - 1) * RESLON + (self.lon * 2  + RESLON - 1) % RESLON]
    value[1] = source[day * GS + ((self.lat + 90) * 2 - 1) * RESLON + (self.lon * 2)]
    value[2] = source[day * GS + ((self.lat + 90) * 2) * RESLON + (self.lon * 2  + RESLON - 1) % RESLON]
    value[3] = source[day * GS + ((self.lat + 90) * 2) * RESLON + (self.lon * 2)]
    countNA = 0
    sumval = 0
    sumw = 0
    for i in range(0, 4):
      if value[i] == NA:
        countNA += 1
      else:
        weight = math.cos(sourceLat[i] * PI / 180.0)
        sumval +=  weight * value[i]
        sumw += weight
    if countNA > 2:
      return (NA, countNA)
    return (sumval / sumw, countNA)
  
  def interpolate(self, tmaxSource, tminSource, precSource):
    for day in range(0, WEEKS4): (self.tmax[day], self.tmaxNA[day]) = self.interpolateDay(day, tmaxSource)
    for day in range(0, WEEKS4): (self.tmin[day], self.tminNA[day]) = self.interpolateDay(day, tminSource)
    for day in range(0, WEEKS4): (self.prec[day], self.precNA[day]) = self.interpolateDay(day, precSource)
    self.prec34 = weeks2sum(self.prec, 0)
    self.prec56 = weeks2sum(self.prec, WEEKS2)
    tmin34 = weeks2sum(self.tmin, 0)
    tmin56 = weeks2sum(self.tmin, WEEKS2)
    tmax34 = weeks2sum(self.tmax, 0)
    tmax56 = weeks2sum(self.tmax, WEEKS2)
    if self.prec34 != NA: self.prec34 /= 10
    else: self.prec34 = float('NAN')
    if self.prec56 != NA: self.prec56 /= 10
    else: self.prec56 = float('NAN')
    self.temp34 = (tmin34 / WEEKS2 + tmax34 / WEEKS2) / 2 if tmin34 != NA and tmax34 != NA else float('NAN')
    self.temp56 = (tmin56 / WEEKS2 + tmax56 / WEEKS2) / 2 if tmin56 != NA and tmax56 != NA else float('NAN')

def isLeap(year):
  return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def monthLength(month, year):
  if month == 2: return 29 if isLeap(year) else 28 
  if month in [4, 6, 9, 11]: return 30
  return 31 

class Date:
  year = 0
  month = 0
  day = 0
  passParsing = False
  parsingErrorMessage = ""
  def getText(self):
    return str(self.year) + ("0" if self.month < 10 else "") + str(self.month) + ("0" if self.day < 10 else "") + str(self.day)
  
  def getText2(self):
    return str(self.year) + "-" + ("0" if self.month < 10 else "") + str(self.month) + "-" + ("0" if self.day < 10 else "") + str(self.day)

  def __init__(self, date):
    if len(date) < 10:
      self.passParsing = False
      self.parsingErrorMessage = "Incorrect (too short) submission deadline!"
      return
    self.year = int(date[0:4])
    self.month = int(date[5:7])
    self.day = int(date[8:10])
    if self.year < 1900 or self.year > 9999:
      self.passParsing = False
      self.parsingErrorMessage = "Incorrect submission deadline, could not parse year!"
      return
    if self.month < 1 or self.month > 12:
      self.passParsing = False
      self.parsingErrorMessage = "Incorrect submission deadline, could not parse month!"
      return
    if self.day < 1 or self.day > 31:
      self.passParsing = False
      self.parsingErrorMessage = "Incorrect submission deadline, could not parse day!"
      return
    if self.day > monthLength(self.month, self.year):
      self.passParsing = False
      self.parsingErrorMessage = "Incorrect submission deadline, such day does not exist in the given month!"
      return
    self.passParsing = True
  
  def correctDay(self):
    while self.day > monthLength(self.month, self.year):
      self.day -= monthLength(self.month, self.year)
      self.month += 1
      if self.month == 13:
        self.month = 1
        self.year += 1

  def dayOfYear(self):
    ans = 0
    for m in range(1, self.month): ans += monthLength(m, self.year)
    return ans + self.day - 1

  def __add__(self, days):
    newDate = copy.deepcopy(self)
    newDate.day += days
    newDate.correctDay()
    return newDate

def checkPath(path):
  if path[-1:] != '/' and path[-1:] != '\\': path += "/"
  return os.path.isdir(path)

def currentYear():
  return int(datetime.today().year)

if __name__ == "__main__": 
  usage = "  Call the program with the following command line arguments:\n    1st argument: submission deadline in YYYY-MM-DD format\n    2nd argument: path to downloaded temperature data\n    3rd argument: path to downladed precipitation data\n"
  argc = len(sys.argv)
  targetDay = Date(sys.argv[1]) if argc > 1 else Date("2019-01-01")
  pathTemp = sys.argv[2] if argc > 2 else "data/temperature/"
  pathPrec = sys.argv[3] if argc > 3 else "data/precipitation/"
  
  if not(targetDay.passParsing):
    print(targetDay.parsingErrorMessage)
    print(usage)
    sys.exit(1)
  if not(checkPath(pathTemp)):
    print("  Incorrect path to temperature data!")
    print(usage)
    sys.exit(1)
  if not(checkPath(pathPrec)):
    print("  Incorrect path to precipitation data!")
    print(usage)
    sys.exit(1)
  nowYear = currentYear()
  firstDay = targetDay + WEEKS2
  lastDay = targetDay + (WEEKS2 + WEEKS4 - 1)
  
  for year in range(firstDay.year, lastDay.year + 1):
    fn = "CPC_GLOBAL_T_V0.x_0.5deg.lnx." + str(year)
    if not(os.path.isfile(pathTemp + fn)):
      if year < nowYear:
        if not(os.path.isfile(pathTemp + fn + ".gz")):
          print("  Downloading temperature data for year " + str(year) + "...\n")
          err = os.system("curl ftp://ftp.cpc.ncep.noaa.gov/precip/PEOPLE/wd52ws/global_temp/" + fn + ".gz --output " + pathTemp + fn + ".gz")
          if err or not(os.path.isfile(pathTemp + fn + ".gz")):
            print("\n  Downloading temperature data for year " + str(year) + " failed!\n")
            break
          print("")
        print("  Extracting temperature data for year " + str(year) + "...\n")
        err = os.system("gzip -dNkv   " + pathTemp + fn + ".gz")
        if err or not(os.path.isfile(pathTemp + fn)):
          print("\n  Extracting temperature data for year " + str(year) + " failed!\n")
          break
        print("")
      elif year == nowYear:
        print("  Downloading temperature data for year " + str(year) + "...\n")
        err = os.system("curl ftp://ftp.cpc.ncep.noaa.gov/precip/PEOPLE/wd52ws/global_temp/" + fn + " --output " + pathTemp + fn)
        if err or not(os.path.isfile(pathTemp + fn)):
          print("\n  Downloading temperature data for year " + str(year) + " failed!\n")
          break
        print("")
      else:
        print("\n  Temperature data for year " + str(year) + " not known!")
        break
  
  for year in range(firstDay.year, lastDay.year + 1):
    if not(os.path.isdir(pathPrec + str(year))):
      err = folderCreate(pathPrec + str(year))
      if err or not(os.path.isdir(pathPrec + str(year))):
        print("  Creating folder " + pathPrec + str(year) + " failed!")
        break 
  
  for day in range(0, WEEKS4):
    date = firstDay + day
    fn = "PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx." + date.getText() + ((".RT" if date.year > 2006 else "RT") if date.year > 2005 else "")
    if not(os.path.isfile(pathPrec + str(date.year) + "/" + fn)):
      if date.year < 2009:
        if not(os.path.isfile(pathPrec + str(date.year) + "/" + fn + ".gz")):
          print("  Downloading precipitation data for date " + date.getText2() + "...\n")
          url = "ftp://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/" + ("V1.0/" if date.year < 2006 else "RT/") + str(date.year) + "/" 
          err = os.system("curl " + url + fn + ".gz --output " + pathPrec + str(date.year) + "/" + fn + ".gz")
          if err or not(os.path.isfile(pathPrec + str(date.year) + "/" + fn + ".gz")):
            print("\n  Downloading precipitation data for date " + date.getText2() + " failed!\n")
            break
          print("")
        print("  Extracting precipitation data for date " + date.getText2() + "...\n")
        err = os.system("gzip -dNkv   " + pathPrec + str(date.year) + "/" + fn + ".gz")
        if err or not(os.path.isfile(pathPrec + str(date.year) + "/" + fn)):
          print("\n  Extracting precipitation data for date " + date.getText2() + " failed!\n")
          break
        print("")
      elif date.year <= nowYear:
        print("  Downloading precipitation data for date " + date.getText2() + "...\n")
        err = os.system("curl ftp://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/RT/" + str(date.year) + "/" + fn + " --output " + pathPrec + str(date.year) + "/" + fn)
        if err or not(os.path.isfile(pathPrec + str(date.year) + "/" + fn)):
          print("\n  Downloading precipitation data for date " + date.getText2() + " failed!\n")
          break
        print("")        
      else:
        print("\n  Precipitation data for date " + date.getText2() + " not known!")
        break
  
  tmax = [NA] * (WEEKS4 * GS)
  tmin = [NA] * (WEEKS4 * GS)
  prec = [NA] * (WEEKS4 * GS)
  tempIntegrity = 0
  
  fn = "CPC_GLOBAL_T_V0.x_0.5deg.lnx." + str(firstDay.year)
  if os.path.isfile(pathTemp + fn):
    checksize = fileSize(pathTemp + fn)
    if checksize != (365 + (1 if isLeap(firstDay.year) else 0)) * 4 * GS * 4:
      print("  File " + pathTemp + fn + " is corrupted! Remove it manually and try to redownload.")
      sys.exit(1) 
    
    infile = open(pathTemp + fn, "rb")
    infile.seek(firstDay.dayOfYear() * GS * 4 * 4)
  
    for day in range(0, WEEKS4):
      date = firstDay + day
      if day > 0 and date.dayOfYear() == 0:
        fn = "CPC_GLOBAL_T_V0.x_0.5deg.lnx." + str(firstDay.year + 1)
        if os.path.isfile(pathTemp + fn):
          infile.close()
          checksize = fileSize(pathTemp + fn)
          if checksize != (365 + (1 if isLeap(firstDay.year + 1) else 0)) * 4 * GS * 4:
            print("  File " + pathTemp + fn + " is corrupted! Remove it manually and try to redownload.")
            sys.exit(1) 
          infile = open(pathTemp + fn, "rb")
        else:
          break
      tmax[day * GS : day * GS + GS] = list(struct.unpack('f'*GS, infile.read(4 * GS)))
      infile.seek(GS * 4, 1)
      tmin[day * GS : day * GS + GS] = list(struct.unpack('f'*GS, infile.read(4 * GS)))
      infile.seek(GS * 4, 1)
      tempIntegrity += 1
    infile.close()
  
  precIntegrity = 0
  for day in range(0, WEEKS4):
    date = firstDay + day
    fn = "PRCP_CU_GAUGE_V1.0GLB_0.50deg.lnx." + date.getText() + ((".RT" if date.year > 2006 else "RT") if date.year > 2005 else "")
    if os.path.isfile(pathPrec + str(date.year) + "/" + fn):
      checksize = fileSize(pathPrec + str(date.year) + "/" + fn)
      if checksize != 2 * GS * 4:
        print("  File " + (pathPrec + str(date.year) + "/" + fn) + " is corrupted! Remove it manually and try to redownload.")
        sys.exit(1) 
      infile = open(pathPrec + str(date.year) + "/" + fn, "rb")
      prec[day * GS : day * GS + GS] = list(struct.unpack('f'*GS, infile.read(4 * GS)))
      infile.close()
      precIntegrity += 1
  
  for day in range(0, WEEKS4):
    date = firstDay + day
    if date.year != nowYear: continue
    isData = False
    for i in range(0, GS):
      if tmax[day * GS + i] != NA:
        isData = True
        break
    if not(isData):
      print("  Temperature data for date " + date.getText2() + " empty. Trying to redownload...\n")
      fn = "CPC_GLOBAL_T_V0.x_0.5deg.lnx." + str(nowYear)
      err = os.system("curl --range " + str(date.dayOfYear() * GS * 4 * 4) + "-" + str((date.dayOfYear() + 1) * GS * 4 * 4 - 1) + " ftp://ftp.cpc.ncep.noaa.gov/precip/PEOPLE/wd52ws/global_temp/" + fn + " --output " + pathTemp + fn + "." + date.getText())
      if err or not(os.path.isfile(pathTemp + fn + "." + date.getText())):
        print("\n  Redownloading temperature data for date " + date.getText2() + " failed!\n")
        break
      print("")
      if os.path.isfile(pathTemp + fn + "." + date.getText()):
        checksize = fileSize(pathTemp + fn + "." + date.getText())
        if checksize == 4 * GS * 4:
          infile = open(pathTemp + fn + "." + date.getText(), "rb")
          daytemp = list(struct.unpack('f'*(4 * GS), infile.read(4 * 4 * GS)))
          infile.close()
          isData = False
          for i in range(0, GS):
            if daytemp[i] != NA:
              isData = True
              break
          if not(isData):
            print("\n  Temperature data for date " + date.getText2() + " still empty. Try again later.\n")
            break
          tmax[day * GS : day * GS + GS] = daytemp[0 : GS]  
          tmin[day * GS : day * GS + GS] = daytemp[2 * GS : 3 * GS]  
          if os.path.isfile(pathTemp + fn):
            print("  Updating " + pathTemp + fn + " with new data...\n")
            io = open(pathTemp + fn, "r+b")  
            io.seek(date.dayOfYear() * GS * 4 * 4)
            io.write(struct.pack('f'*(4 * GS), *daytemp))
            io.close()
        else:
          print("\n  Redownloading temperature data for date " + date.getText2() + " failed!\n")
          break
      else:
        print("\n  Redownloading temperature data for date " + date.getText2() + " failed!\n")
        break

  infile = open("../data/target_points.csv")
  lines = infile.readlines()[1:]
  targetPoints = [PlaceData(lines[i]) for i in range(0, 514)] 
  infile.close()
  
  out = open("gt_" + targetDay.getText2() + ".csv", "w")
  print("lat,lon,temp34,prec34,temp56,prec56", file = out)
  for point in targetPoints:
    point.interpolate(tmax, tmin, prec)
    print(point.lat, point.lon, "{0:.6g}".format(point.temp34), "{0:.6g}".format(point.prec34), "{0:.6g}".format(point.temp56), "{0:.6g}".format(point.prec56), sep = ",", file = out)
  out.close()

