import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
import os
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D


def open_profilo(folder):
    """
    Opens the raw data from the profilometer
    dat : (n, 4)  float30 array
        Altitude; Intensity; Buffer time (in ms)
    pos : (m, 4)  float32 array
        X; Y; Buffer time (in ms)
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
    reference : use to specify the two reference points on which are computed the tendency
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
        #then orthogonal to the rows
        start = np.average(data_per_row[0][:,0])
        end = np.average(data_per_row[-1][:,0])
        for j in range(n_rows):
            data_per_row[j][:,0]=data_per_row[j][:,0]-start-(end-start)*j/n_rows
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

def profile_plot(positions_per_row,data_per_row,positions_interpolations,row_spacing=0,mult=1,**kwargs):
    """
    Plots the result of the data analysis.
    row_spacing : shifts the successive rows for lisibility
    mult : multiplies the result by a constant (useful if you used the wrong pointer)
    """
    for j in range(len(positions_per_row)):
        x = positions_interpolations[j](data_per_row[j][:,2])
        y = positions_per_row[j][0,1]- positions_per_row[0][0,1]
        h = mult*data_per_row[j][:,0]
        plt.plot(x,h+j*row_spacing,label='y={:10.2f} mm'.format(y),**kwargs)
    plt.legend()
    plt.grid(which='both')
    return(x,h)

def substract_trend(calib,data_per_row,positions_interpolations):
    """
    Substract the trend in the folder `calib`
    """
    a=calibration_deviation_profilo(calib)
    for j in range(len(data_per_row)):
        x = positions_interpolations[j](data_per_row[j][:,2])
        data_per_row[j][:,0] = data_per_row[j][:,0]-a(x)
    return(data_per_row)


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



def fully_open_data(location,calibration=None):
    """
    Functionnalization of all the work done when opening the file.
    Input :
        location :
            the location of the file you want to use
        calibration
            the location of the Altitude calibration to use
            if None, no calibration is done
    output :
        positions_per_row :
            list, containing arrays describing position
            Each array corresponds to one measurement row
            format of each array:
                   [[X(t_0), Y(t_0), t_0 (ms)],
                    [X(t_1), Y(t_1), t_1 (ms),
                    ...,
                    [X(t_m), Y(t_m), t_m (ms)]], dtype=float32)
        data_per_row
            list, containing arrays describing ALtitude and Intensity
            Each array corresponds to one measurement row
            format of each array:
                   [[A(t'_0), I(t'_0), t'_0 (ms)],
                    [1(t'_1), I(t'_1), t'_1 (ms),
                    ...,
                    [A(t'_n), I(t'_n), t'_n (ms)]], dtype=float32)
        positions_interpolations
            list, containing <scipy.interp1d>
            Each <scipy.interp1d> corresponds to a function that interpolates
            the position of data_per_row from its time.
    """
    # this line is used to fix a bug in the software
    correct_description(location)

    # open the data
    full_info=open_profilo(location)
    data=full_info[0]
    positions=full_info[1]
    description=full_info[2]

    # Apply proper cleaning (some may not apply to your usecase)

    # Interpolate profilometer times
    data=interpol_time(data,description)

    # Crop the zeros, due to bad reading of the profilometer
    data=crop_data_clean(data)


    # Separate data in rows
    positions_per_row=separate_rows(positions)
    data_per_row=separate_rows(data)

    # Interpolate positions of Zaber
    # Clean positions out of interpolation range
    data_per_row=clean_data_interpol(data_per_row,positions_per_row)

    #this line doesn't work...
    #data_per_row=outliars_removal(data_per_row,width=0.2)

    # Interpolate
    positions_interpolations = full_position_interpolation(positions_per_row)

    # Remove the variations due to the Zaber translation plates
    if calibration:
        data_per_row = substract_trend(calibration,data_per_row,positions_interpolations)

    return(positions_per_row,data_per_row,positions_interpolations)





## smoothing
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def calibration_deviation_profilo(location, plot=False):
    """
    Location must contain a folder, containing several acquisitions that will be
    used as a reference to make a proper zero for the altitude axis.
    ! Only the last row of every acquisition will be taken into account !

    output : np.poly1d that fits x
    """
    # open the data
    xs=[]
    hs=[]
    for file in os.listdir(location):
        positions_per_row,data_per_row,positions_interpolations=fully_open_data(location+file)
        data_per_row=tendency_removal(data_per_row,positions_interpolations,reference=100)
        data_per_row=tendency_removal_per_row(data_per_row,positions_interpolations,reference=100)
        for j in range(len(positions_interpolations)):
            x = positions_interpolations[j](data_per_row[j][:,2])
            h = data_per_row[j][:,0]
            xs.append(x)
            hs.append(h)
    # clean the data format
    xtot=[]
    htot=[]
    for i in range(len(xs)):
        xtot=xtot+list(xs[i])
        htot=htot+list(hs[i])
    # sort x
    zipped_lists = zip(xtot, htot)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    xtot, htot= [ list(tuple) for tuple in  tuples]
    xtot=np.array(xtot)
    htot=np.array(htot)
    # make a mean befor fitting
    mean_x_axis = [i for i in range(int(max([max(j) for j in xs])))]
    ys_interp = [np.interp(mean_x_axis, xs[i], hs[i]) for i in range(len(xs))]
    mean_y_axis = np.mean(ys_interp, axis=0)
    # fit
    a=np.poly1d(np.polyfit(mean_x_axis,mean_y_axis,9))
    # plot to check
    if plot:
        htotfit=a(xtot)
        plt.scatter(xtot,htot)
        plt.plot(mean_x_axis, mean_y_axis,color="orange")
        yhat=savitzky_golay(htotfit,51,3)
        plt.plot(xtot,yhat,color="red")
        plt.show()
    return(a)



### Profile plot in 3D


def x_interpol_i(x,positions_per_row_i,data_per_row_i,positions_interpolations_i):
    """
    interpolates the positions with a cubic function
    x : positions to compute
    ..._i : element of ...
    output : f(x)
    """
    x_reel = positions_interpolations_i(data_per_row_i[:,2])
    f_interp = interpolate.interp1d(x_reel, data_per_row_i[:,0], kind='linear')
    return(f_interp(x))

def x_interpol(step,positions_per_row, data_per_row, positions_interpolations):
    """
    Interpolates the data to put it on a grid
    step = interpolation step (mm)
    """
    y=[]
    h=[]
    x_reel = positions_interpolations[0](data_per_row[0][:,2])
    x=np.arange(x_reel[1],x_reel[-2],step)
    for i in range(len(positions_per_row)):
        y.append(positions_per_row[i][0,1]- positions_per_row[0][0,1])
        h.append(x_interpol_i(x,positions_per_row[i],data_per_row[i],positions_interpolations[i]))
    return(x,np.array(y),np.array(h))


def profile_plot_3D_interpol(step,positions_per_row, data_per_row, positions_interpolations,mult=1):
    """
    Plots a 3D profile with interpolation
    step = interpolation step (mm)
    """
    x,y,h=x_interpol(1,positions_per_row, data_per_row, positions_interpolations)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X, Y = np.meshgrid(x, y)
    Z = h

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return(fig,ax,surf)

def profile_plot_3D_meshed(x,y,h):
    """
    Plots a meshed surface (with the same Xs for all data)
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X, Y = np.meshgrid(x, y)
    Z = h

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=True)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return(fig,ax,surf)

def profile_plot_3D(positions_per_row, data_per_row, positions_interpolations, mult=1):
    """
    Plots the result of the data analysis.
    mult : multiplies the result by a constant (useful if you used the wrong pointer)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    #ax.set(xlim=(0, 150), ylim=(0, 150))
    X=[]
    Y=[]
    H=[]
    for j in range(len(positions_per_row)):
        x = positions_interpolations[j](data_per_row[j][:,2])
        y = positions_per_row[j][0,1]- positions_per_row[0][0,1]
        h = mult*data_per_row[j][:,0]
        for xi in x:
            X.append(xi)
            Y.append(y)
        for hi in h:
            H.append(hi)
    surf = ax.plot_trisurf(X, Y, H, cmap=cm.coolwarm,antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return(fig,ax,surf)



