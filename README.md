# Asteroid Chemical Composition AI

C, S, and M are abbreviation of the 3 major asteroid classifications (Carbonaceous, Silicate, Metallic)

Import any asteroid spectral reflectance data (With consistent datapoints between wavelengths of 0.8 - 2.4 micrometers), then normalise the data at a value of 1 micrometer and interpolate it using:

--from scipy import interpolate

--f = interpolate.interp1d(wavelengthlist0, reflectancelist0)
--meanreflectance = f(np.arange(0.85, 2.4, 0.0025))


There are a total of 33 materials considered here with the average carbonaceous and silicate composition (Total 35 possible constituents), feel free to add more but changes will need to be done.

See Metalsn to see all of the possible constituents considered, the format of the neural network follows the format of "AllValues.csv" (Once generated)

Download the spectral reflectance data of asteroids, then follow the instructions once running XMLSorter.py, Asteroid-Mean-Reflectance.py, RefractiveIndexToSpectralRelfectance.py, and AI.py in respective order.

Please feel free to reach out to amaurypat@gmail.com if you have any problems with using the AI!
 
## Libraries

- PDS4 Tools (v1.3)

### Python Libraries

- pytorch
- numpy (version < 2)
- scipy
- matplotlib (Optional, to visualise data)
- os
- datetime