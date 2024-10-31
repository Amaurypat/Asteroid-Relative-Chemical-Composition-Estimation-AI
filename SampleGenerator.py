import os
import numpy as np
import csv
import math
import random

cwd = os.getcwd()
os.chdir('Metalsf')
filelist = os.listdir()
filelist.remove('Allvalues.csv')
filelist.remove('slist.txt')
filelist.remove('Clist.txt')
filelist.remove('Averaged')
print(filelist)
slist = np.loadtxt('slist.txt')
clist = np.loadtxt('Clist.txt')
allDict = {}
allval = []
allname = []
sampleDict = {}
sample = np.zeros(shape=(40, 620))
nrow = ''
smpsum = np.zeros(shape=(1, 620))
comp = np.zeros(shape=(1, len(filelist)))
with open('AllValues.csv', 'r') as file:
    reader = csv.reader(file)
    temp1 = next(reader)
    temp2 = [eval(i) for i in next(reader)]
    for i in range(0,len(filelist)-1):
        allname.append((str(temp1[i].strip('][').split(', '))).strip("['']"))
        allval.append(temp2[i])
npname = np.array(allname)
npval = np.array(allval)
n = len(filelist)
nsum = 0
c = 0
s = 0
os.chdir(cwd)
for g in range(0, 10000):
    nsum = random.randint(0,100000)
    result = []
    tot = 0
    c = random.randint(0,100000 - nsum)
    s = 100000 - (nsum + c)
    for i in range(n):
        print(i)
        result.append(random.randint(0, nsum - tot))
        print(result)
        tot = result[i] + tot
    random.shuffle(result)
    for i in range(nsum - sum(result)): 
        result[random.randint(0,3)] += 1
    for i in range (0, len(filelist)-1):
        sample[i] = npval[i]*result[i]*0.00001*((npval[i][200])**-1)
        smpsum = smpsum + sample[i]
    smpsum = smpsum + c*0.00001*(clist*((clist[200])**-1))
    smpsum = smpsum + s*0.00001*(slist*((slist[200])**-1))
    smpsum = smpsum*(smpsum[0][200])**-1
    result.append(c)
    result.append(s)
    np.savetxt('samples/sample' + str(g) + '.txt', np.array(smpsum))
    np.savetxt('samples/comp' + str(g) + '.txt', np.array(result)*0.00001)


