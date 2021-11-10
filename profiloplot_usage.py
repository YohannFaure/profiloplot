#Typical usage of profiloplot.py

## Import profiloplot
#from profiloplot import *
exec(open("C:/Users/User/Desktop/profiloplot.py").read())

## Import useful modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

## Open the files
location="C:/Users/User/Documents/Acquisition backups/2021-11-10  T  10-09-52/"
correct_description(location)
full_info=open_profilo(location)
data=full_info[0]
positions=full_info[1]
description=full_info[2]

## Crop the zeros
data=crop_data_clean(data)

## Interpolate profilometer times
data=interpol_time(data,description)

## Separate data in rows
positions_per_row=separate_rows(positions)
data_per_row=separate_rows(data)

## Interpolate positions of Zaber
# Clean positions out of interpolation range
data_per_row=clean_data_interpol(data_per_row,positions_per_row)
# Interpolate
positions_interpolations = full_position_interpolation(positions_per_row)

## Remove linear tendancy
# Global
data_per_row=tendency_removal(data_per_row,positions_interpolations,reference=100)
# Different for each row (removes the profile in y axis)
# data_per_row=tendency_removal(data_per_row,positions_interpolations,reference=100)

## Plot
profile_plot(positions_per_row,data_per_row,positions_interpolations,row_spacing=0,mult=1)
plt.x_label()
plt.show()














