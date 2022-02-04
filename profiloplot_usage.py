#Typical usage of profiloplot.py
## Import useful modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#from profiloplot import *
exec(open("/home/yohann/Desktop/profilo/profiloplot-main/profiloplot.py").read())


## Open the files
#location="C:/Users/User/Desktop/bloc34_polis/"
location="/home/yohann/Desktop/profilo/calibprofilo/profile1/"
calibration="/home/yohann/Desktop/profilo/calibprofilo/"


help(fully_open_data)
positions_per_row,data_per_row,positions_interpolations=fully_open_data(location,calibration=calibration)

## Remove linear x-tendacy
data_per_row=tendency_removal(data_per_row,positions_interpolations,reference=100)

# Different for each row (removes the profile in y axis)
#data_per_row=tendency_removal_per_row(data_per_row,positions_interpolations,reference=100)





## Plot
profile_plot(positions_per_row,data_per_row,positions_interpolations,row_spacing=0,mult=1)
plt.xlabel("x axis (mm)")
plt.ylabel("Altitude (µm)")

plt.show()














