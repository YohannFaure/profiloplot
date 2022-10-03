#Typical usage of profiloplot.py
## Import useful modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#from profiloplot import *
exec(open("C:/Users/User/Desktop/profiloplot/profiloplot.py").read())


## Open the files
location="C:/Users/User/Desktop/profil_blocs/bloc1_p5_propre/"
calibration="C:/Users/User/Desktop/profiloplot/calibprofilo/"


#help(fully_open_data)
positions_per_row,data_per_row,positions_interpolations=fully_open_data(location,calibration=calibration)

## Remove linear x-tendency (and y-tendency if need be)
data_per_row=tendency_removal(data_per_row,positions_interpolations)#,remove_orthogonal=True)



# Different for each row (removes the profile in y axis)
#data_per_row=tendency_removal_per_row(data_per_row,positions_interpolations,reference=100)



## Plot


profile_plot(positions_per_row,data_per_row,positions_interpolations,row_spacing=0,mult=1,linewidth=0,marker=".")
plt.xlabel("x axis (mm)")
plt.ylabel("Altitude (µm)")
plt.show()



## Two types of 3D plot

### Interpolated (less computational power needed)
# Interpolation step (to create a single X-axis for all lines)
step=0.01 #mm

fig,ax,surf = profile_plot_3D_interpol(step, positions_per_row, data_per_row, positions_interpolations)

fig.show()

### Full polygon plot (this might crash, very power hungry)

fig,ax,surf = profile_plot_3D(positions_per_row, data_per_row, positions_interpolations)

fig.show()











