import os
import numpy as np
import requests
import shutil
cwd = os.getcwd()
folderdirectory = input("Enter your folder directory: ")
print("Sorting files at: " + folderdirectory)
os.chdir(folderdirectory)
sstr = ''
ident = ''
tabfile = ''
response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=Ceres&phys-par=1")
filelist = os.listdir()
os.chdir(cwd)
for val, file in enumerate(filelist) : 
    if file.endswith(".xml"):
        tabfile = file.partition('.xml')[0] + '.csv'
        sstr = file.partition('_')[0]
        response = requests.get("https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=" + sstr + "&phys-par=1")
        for attrs in response.json()['phys_par']:
            if attrs['desc'] == 'Tholen spectral taxonomic classification' or 'SMASSII spectral taxonomic classification':
                ident = attrs['value']
            if ident == 'S':
                try:
                    shutil.move(folderdirectory + '/' + file, 'S')
                    shutil.move(folderdirectory + '/' + tabfile, 'S')
                except:
                    print("Already Moved")
                print('wowy')
            if ident == "C":
                try:
                    shutil.move(folderdirectory + '/' + file, 'C')
                    shutil.move(folderdirectory + '/' + tabfile, 'C')
                except:
                    print("Already Moved")
                print('wowy')
            if ident == "M":
                try:
                    shutil.move(folderdirectory + '/' + file, 'M')
                    shutil.move(folderdirectory + '/' + tabfile, 'M')
                except:
                    print("Already Moved")
                print('wowy')            
        else:
            print('Nothing found!')




