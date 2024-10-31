import csv
import os
from scipy import interpolate
import numpy as np
##import matplotlib

csv_file_path = ''
newcsv_file_path = 'Metalsr'
wvlist = []
nlist = []
fnlist = []
nplist = []
rlist = []
allDict = {}
fails = -1
wlc = 0
k = 1
cwd = os.getcwd()
os.chdir('Metalsn/')
filelist = os.listdir()
for nfile in range(0,len(filelist)):
    os.chdir(cwd)
    os.chdir('Metalsn')
    csv_file_path = filelist[nfile]
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                if row[0] == 'wl':
                    wlc = wlc + 1
                    if wlc == 2:
                        wlc = 0
                        break
                    continue
                if row[1] == 'k':
                    continue
            except:
                fails = fails + 1
            try:
                wvlist.append(float(row[0]))
                nlist.append(float(row[1]))
            except:
                fails = fails + 1
    rlist = [(((1-x)*((1+x)**-1))**2) for x in nlist]      
    try:
        f = interpolate.interp1d(wvlist, rlist, kind='cubic', bounds_error=False, fill_value='extrapolate')
    except:
        print('wvlist: ' + str(wvlist))
        print('nlist: ' + str(rlist))
        exit()
    allDict[csv_file_path.replace('.csv', '')] = (f(np.arange(0.85, 2.4, 0.0025))).tolist()
    os.chdir(cwd)
    os.chdir('Metalsr/')
    np.savetxt(csv_file_path, f(np.arange(0.85, 2.4, 0.0025)))
    with open("Allvalues.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=allDict.keys())
        writer.writeheader()
        writer.writerow(allDict)
    wvlist.clear()
    nlist.clear()
    rlist.clear()
    wlc = 0
    k = 0

print("Fails: " + str(fails))
print(allDict)