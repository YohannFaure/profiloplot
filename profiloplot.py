import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def open_profilo(folder):
    """
    Opens the raw data from the profilometer
    dat : (n, 4)  float30 array
    pos : (m, 4)  float32 array
    des : (14, 2) str array
    """
    if folder[-1]!="/":
        folder+="/"
    dat=np.loadtxt(folder+'data.txt', dtype=np.float32, delimiter=';')
    pos=np.loadtxt(folder+'positions.txt', dtype=np.float32, delimiter=';')
    des=np.loadtxt(folder+'description.txt', dtype=str, delimiter=':')
    return(dat,pos,des)


def correct_description(folder):
    """
    Corrects an early bug in the profilo software.
    might become obsolete very soon.
    """
    if folder[-1]!="/":
        folder+="/"
    try :
        des=np.loadtxt(folder+'description.txt', dtype=str, delimiter=':')
    except:
        f=open(folder+"description.txt",'r+')
        old=f.read()
        f.close()
        if not ':' in old[440:460]:
            lst=list(old)
            lst[449]=': '
            new=''.join(lst)
            f=open(folder+"description.txt",'w')
            _=f.write(new)
            f.close()
    return(None)

## clean data


def crop_data_clean(data):
    """
    This simply removes all data that's outside of the sample (data with a 0 measurement)
    This does not affect timings
    """
    temp = data[np.where(data[:,0]!=0)]
    return(temp)

## interpolate times

def interpol_time(data,description):
    """
    Interpolation of the time of the datapoints from the profilometer
    this data comes in batches of up to 10 points (10 when operating normaly, less when switching rows)
    """
    #I don't want to modify the original data
    data_cop=np.copy(data)
    time = data_cop[:,2]
    row = data_cop[:,3]
    freq = int(description[8,1])
    delta_t=1000/freq
    # determine row change
    change=[0]+list(np.where(row[:-1] != row[1:])[0]+1)+[len(row)]
    # add time
    for i in range(1,len(change)):
        time[change[i-1]:change[i]]=time[change[i-1]]+np.arange(0,change[i]-change[i-1])*delta_t
    #old method
    # for i in range(1,len(time)):
    #     #if no change in row, we can simply add delta_t, else, time restarts
    #     if row[i]==row[i-1]:
    #         time[i]=time[i-1]+delta_t
    return(data_cop)

## interpolate positions from Zaber Data

def interpol_position(positions):
    """
    interpolates the position of each row
    interp_x_t_row : array of functions x_i(t), with i the number of the row
    LEGACY.
    """
    pos=positions
    n_rows = int(np.max(dat[:,-1]))
    pos_per_row=[pos[np.where(pos[:,3]==i)] for i in range(1,n_rows+1)]
    pos_per_row=[row[:,:3] for row in pos_per_row]
    y_row=[row[0,1] for row in pos_per_row]
    x_row=[row[:,0] for row in pos_per_row]
    t_row=[row[:,2] for row in pos_per_row]
    interp_x_t_row = [interpolate.interp1d(t_row[k],x_row[k]) for k in range(n_rows)]
    return(interp_x_t_row)

def separate_rows(position_or_data):
    """
    separates the data/positions in rows
    dat_per_row : n_rows list containing (n,4) arrays for each row
    """
    dat=position_or_data
    n_rows = int(np.max(dat[:,-1]))
    dat_per_row=[dat[np.where(dat[:,3]==i)] for i in range(1,n_rows+1)]
    dat_per_row=[row[:,:3] for row in dat_per_row]
    return(dat_per_row)

def interpol_position_row(positions_row):
    """
    Interpolates the position of the data in a single row
    interp_x_t : function x(t) interpolating zaber data.
    Usage : interp_x_t(t) returns a position associated with time t
    """
    interp_x_t = interpolate.interp1d(positions_row[:,2],positions_row[:,0])
    return(interp_x_t)


def clean_data_interpol(data_per_row,positions_per_row):
    """
    removes all data outside of the interpolation range
    """
    #just a trick to copy without mofification
    dat_cop=[np.copy(row) for row in data_per_row]
    dat_cop = [ dat_cop[k][np.where( (dat_cop[k][:,-1]<np.max(positions_per_row[k][:,-1]) ) )] for k in range(len(dat_cop))]
    dat_cop =[ dat_cop[k][np.where( (dat_cop[k][:,-1]>np.min(positions_per_row[k][:,-1])) )] for k in range(len(dat_cop))]
    return(dat_cop)


def full_position_interpolation(positions_per_row):
    """
    Gives a list of interpolation functions for all rows of a data
    output : [x_1, ..., x_n] with n the number of rows and x_i a function of t
    """
    return([interpol_position_row(row) for row in positions_per_row])


## remove tendency


def tendency_removal(data_per_row_in,position_interpolation,reference=100,reference_size=20,remove_orthogonal=True):
    """
    Removes the global tendency for a full experiment (same correction to all rows)
    remove_orthogonal : remove a potential tilt in orthogonal direction to the rows
    """
    data_per_row=[np.copy(row) for row in data_per_row_in]
    #first along a row
    pi=position_interpolation
    start=[]
    end=[]
    n_rows=len(data_per_row)
    for j in range(n_rows):
        k = data_per_row[j]
        start.append([pi[j](k[reference,-1]),np.average(k[reference-reference_size:reference+reference_size,0])])
        end.append([pi[j](k[-reference,-1]),k[-reference,0]])
    start=np.array(start)
    end=np.array(end)
    avg_start=np.average(start,axis=0)
    avg_end=np.average(end,axis=0)
    for j in range(n_rows):
        data_per_row[j][:,0]=data_per_row[j][:,0]-avg_start[1]- (( pi[j](data_per_row[j][:,2]) - avg_start[0] )/(avg_end[0]-avg_start[0]))*(avg_end[1]-avg_start[1])
    if remove_orthogonal:
        #then orthogonal to the row
        start-=avg_start
        end-=avg_end
        for j in range(n_rows):
            data_per_row[j][:,0]=data_per_row[j][:,0]-start[0,1]-(start[-1,1]-start[0,1])*j/n_rows
    return(data_per_row)



def tendency_removal_per_row(data_per_row_in,position_interpolation,reference=100,reference_size=10):
    """
    Removes the tendency for each row, independantly of the others !
    Meaning this function is oblivious to many potential large scale defects of the surface.
    """
    data_per_row=[np.copy(row) for row in data_per_row_in]
    pi=position_interpolation
    start=[]
    end=[]
    n_rows=len(data_per_row)
    for j in range(n_rows):
        k = data_per_row[j]
        start.append([pi[j](k[reference,-1]),np.average(k[reference-reference_size:reference+reference_size,0])])
        end.append([pi[j](k[-reference,-1]),k[-reference,0]])
    start=np.array(start)
    end=np.array(end)
    for j in range(n_rows):
        data_per_row[j][:,0]=data_per_row[j][:,0]-start[j][1]- (( pi[j](data_per_row[j][:,2]) - start[j][0] )/(end[j][0]-start[j][0]))*(end[j][1]-start[j][1])
    return(data_per_row)



## remove outliars


def outliars_removal(data_per_row,width=5):
    """
    Remove elements that do not make sense
    doesn't seem to work...
    """
    new_list=[]
    for i in range(len(data_per_row)):
        rem=[]
        for j in range(1,len(data_per_row[i])-1):
            left_jump =abs(data_per_row[i][j,0]-data_per_row[i][j-1,0])
            right_jump=abs(data_per_row[i][j,0]-data_per_row[i][j+1,0])
            if right_jump>width and left_jump<width :
                rem.append(j)
        keep=np.array( [k for k in range(len(data_per_row[i])) if not k in rem] )
        new_list.append(data_per_row[i][keep])
    return(new_list)





## reconstruct profile (position)

def profile_plot(positions_per_row,data_per_row,positions_interpolations,row_spacing=0,mult=1):
    """
    Plots the result of the data analysis.
    row_spacing : shifts the successive rows for lisibility
    mult : multiplies the result by a constant (useful if you used the wrong pointer)
    """
    for j in range(len(positions_per_row)):
        x = positions_interpolations[j](data_per_row[j][:,2])
        y = positions_per_row[j][0,1]- positions_per_row[0][0,1]
        h = mult*data_per_row[j][:,0]
        plt.plot(x,h+j*row_spacing,label='y={:10.2f} mm'.format(y))
    plt.legend()
    plt.grid(which='both')
    return(None)



## Profile average

def profile_average(positions_per_row,data_per_row_in,positions_interpolations,nbins=1000):
    data_per_row=[np.copy(row) for row in data_per_row_in]
    for j in range(len(positions_interpolations)):
        data_per_row[j][:,-1]=positions_interpolations[j](data_per_row[j][:,-1])
    dat_tot=np.concatenate(data_per_row)
    max_pos=np.max(dat_tot[:,-1])
    min_pos=np.min(dat_tot[:,-1])
    bins=np.arange(min_pos,max_pos,(max_pos-min_pos)/nbins)
    test = np.digitize(dat_tot[:,-1],bins)
    bins_h=np.array([np.average(np.concatenate([dat_tot[np.where(test==j),0][0],np.array([0,0])])) for j in range(0,nbins)])
    return(bins,bins_h)


def average_profile_plot(positions_per_row,data_per_row,positions_interpolations,nbins=1000,shift=0,label=None):
    bins,bins_h=profile_average(positions_per_row,data_per_row,positions_interpolations,nbins)
    plt.plot(bins,bins_h+shift,label=label)
    return(None)



