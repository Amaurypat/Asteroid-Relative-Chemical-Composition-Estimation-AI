import pds4_tools
import os
import numpy as np
##import matplotlib
from datetime import datetime
from scipy import interpolate

cwd = os.getcwd()
folderdirectory = input("Enter your folder directory: ")
os.chdir(folderdirectory)
filelist = os.listdir()
goodfilelist = []
nfails = 0
for file in filelist : 
    if file.endswith(".xml"):
        goodfilelist.append(file)
structures = pds4_tools.read(goodfilelist[0])
try:
    wavelengthlist0 = (structures['table']['WAVELENGTH'])
    reflectancelist0 = (structures['table']['REFLECTANCE'])
except:
    wavelengthlist0 = (structures['TABLE']['Wavelength'])
    reflectancelist0 = (structures['TABLE']['Reflectance'])
f = interpolate.interp1d(wavelengthlist0, reflectancelist0)
meanreflectance = f(np.arange(0.85, 2.4, 0.0025))

for nfile in range(1,len(goodfilelist)) :
    structures = pds4_tools.read(goodfilelist[nfile])
    try:
        wavelengthlist = (structures['TABLE']['Wavelength'])
        reflectancelist = (structures['TABLE']['Reflectance'])
    except:
        wavelengthlist = (structures['table']['WAVELENGTH'])
        reflectancelist = (structures['table']['REFLECTANCE'])
    try:
        f = interpolate.interp1d(wavelengthlist, reflectancelist)
        interpreflectance = f(np.arange(0.85, 2.4, 0.0025))
        meanreflectance = meanreflectance*0.5 + interpreflectance*0.5
    except:
        nfails = nfails + 1
        print('failed')
CorS = ":)"
while CorS != "C" and CorS != "S":
    CorS = input("What type of asteroids is this? [C/S] ")
file_path = CorS + "/" + folderdirectory + "list.txt"
os.chdir(cwd)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
mrs = np.array(meanreflectance)
np.savetxt(file_path, mrs)
print(mrs)
print(f"File '{file_path}' created successfully.")
print('Number of fails: %d' % nfails)

