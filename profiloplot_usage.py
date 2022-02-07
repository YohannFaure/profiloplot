#Typical usage of profiloplot.py
## Import useful modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#from profiloplot import *
exec(open("C:/Users/User/Desktop/profiloplot/profiloplot.py").read())


## Open the files
location="C:/Users/User/Desktop/bloc1_p6_propre/"
calibration="C:/Users/User/Desktop/profiloplot/calibprofilo/"


#help(fully_open_data)
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














