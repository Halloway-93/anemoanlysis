
def pause():
    x = input("Press the <ENTER> key to continue...")
    print(x)


def signaltonoise(a, axis=0, ddof=0):
    import numpy as np
    
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)



def longestNanRun(sequence):
    import numpy as np

    nan_run = np.diff(np.concatenate(([-1], np.where(~np.isnan(sequence))[0], [len(sequence)])))
    nan_seq = np.where(nan_run>1)[0]

    nan_run = nan_run[nan_seq]

    seqs = np.split(nan_seq, np.where(np.diff(nan_seq) != 1)[0]+1)
    final_nan_run = []
    for seq in seqs:
        idx = np.searchsorted(nan_seq,seq)
        final_nan_run.append(sum(nan_run[idx]))

    return max(final_nan_run, default=0)

# got nan_helper from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    import numpy as np

    return np.isnan(y), lambda z: z.nonzero()[0]


def plotFig(trial_idx, tg_dir_h, time_x, vel_x, equ_x, start_a_x, end_a_x, lat_x, trialType='', ax=None, show=False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    
    tg_time = time_x[time_x>=0]
    tg_vel  = np.ones((len(tg_time),1)) * 11
    tg_dir_h = np.ones((len(tg_time),1)) * tg_dir_h

    time_all = np.arange(-200,time_x[-1],1)
    box_x = (time_all > start_a_x) & (time_all < end_a_x)
    
    if 'Red' in trialType:
        tColor = np.array([255, 35, 0])/255
    elif 'Green' in trialType:
        tColor = np.array([0, 255, 35])/255
    else:
        tColor = np.array([0, 0, 0])

    if ax == None:
        f = plt.figure(figsize=(7,4))
        
    plt.suptitle('Trial %d' % trial_idx)
    plt.plot(tg_time, 11*tg_dir_h, linewidth = 1, linestyle = '--', color = '0.5')
    plt.plot(time_x, vel_x, color = tColor)
    plt.plot(time_x, equ_x, color = np.array([0, 0, 230])/255)
    plt.axvline(x = 0, linewidth = 1, linestyle = '--', color = 'k')
    plt.fill_between(time_all, -40, 40, where=box_x, color='gray', alpha=0.1, interpolate=True)
    plt.text(start_a_x, 20, 'onset anticip.', ha='right', va='top', rotation=90)
    plt.text(end_a_x, 20, 'offset anticip.', ha='right', va='top', rotation=90)
    plt.axvline(x = lat_x, linewidth = 1, linestyle = '--', color = 'red')
    plt.text(lat_x, 20, 'latency', ha='right', va='top', rotation=90)
    plt.xlim(-200,600)
    plt.ylim(-25,25)
    plt.xlabel('Time (ms)')
    plt.ylabel('Velocity (deg/s) x axis')
    
    if show:
        plt.show()




def closest(lst, K): 
    import numpy as np

    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return lst[idx], idx


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    g: figure
    fac: factor to adjust
    """
    import numpy as np
    from matplotlib.patches import PathPatch

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def plotBoxDispersion(data, by:str, between:str, groups=None, groupsNames=None, ax=None, jitter=.125, scatterSize:int=.5, showfliers:bool=True, alpha:int=10, showKde:bool=True, showBox:bool=True, cmap=None) -> None:
    '''
    ----------------------------------
    Created by Cristiano Azarias, 2020
    ----------------------------------
    Adapted by Vanessa
    ----------------------------------
    data: data to plot
    by: (list of) variable(s) to group data
    between: (list of) variable(s) to return for grouped data
    alpha: integer to scale the amplitude of the kde dist

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import traceback

    if ax is None: # Check wether an axes was assigned or not
        fig, ax = plt.subplots() # Create figure
        fig.set_size_inches((6,6)) # Set a size
    
    group_by = data.groupby(by)[between] # Create group generator
    if groups is None:
        n_groups = len(group_by.groups)
    else:
        n_groups = len(groups)

    if cmap is None:
        cmap = plt.get_cmap('winter') #timing_cmap() # Load colormap
        colors = cmap(np.linspace(0, 1, n_groups+1)) # Set colors based on the group length
    else:
        colors = cmap
        
    colors50 = np.copy(colors) # Colors copy
    colors50[:,-1] = .5 # Decrease opacity
    colors80 = np.copy(colors)
    colors80[:,-1] = .65 # Decrease opacity

    pos = np.arange(n_groups)+1
    
    if groups is None:
        for idx, group in enumerate(group_by):
            if showBox:
                bplot = ax.boxplot(group[1], positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers) # Plot boxplot
            kde = stats.gaussian_kde(group[1]) # Fit gaussian kde
            x = np.linspace(min(group[1]), max(group[1]), 1000) # Assing x's
            n = len(group[1]) # Assign the number of items per group
            amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
            disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
            ax.scatter(pos[idx]*np.ones(n)+disp, group[1], s=scatterSize, facecolor=colors50[pos[idx]-1], zorder=1) # Plot all data on the right side of boxplot
            if showKde:
                ax.fill_betweenx(x, pos[idx]-kde(x)*amp, pos[idx], facecolor=colors80[pos[idx]-1], zorder=1) # Plot the kde curve on the left side of boxplot
            if showBox:
                for patch in bplot['boxes']: 
                    patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                 patch.set_edgecolor(colors[pos[idx]-1]) # Set boxplot edgecolor to black
                    patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                for patch in bplot['medians']: 
                    patch.set_color('gold') # Set boxplot median to dark yellow
                    patch.set_linewidth(2) # Set boxplot median line width to 2
                    
        ax.set_xticklabels(group_by.groups)
        plt.ylabel(between)
    else:
        for g in groups:
            try: 
                grouptmp = group_by.get_group(g)
                group = grouptmp.dropna()
                idx   = groups.index(g) 

                if showBox:
                    bplot = plt.boxplot(group, positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers) # Plot boxplot
                kde = stats.gaussian_kde(group) # Fit gaussian kde
                x = np.linspace(min(group), max(group), 1000) # Assign x's
                n = len(group) # Assign the number of items per group
                amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
                disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
                plt.scatter(pos[idx]*np.ones(n)+disp, group, s=scatterSize, facecolor=colors50[pos[idx]-1], zorder=1) # Plot all data on the right side of boxplot
                if showKde:
                    plt.fill_betweenx(x, pos[idx]-kde(x)*amp, pos[idx], facecolor=colors80[pos[idx]-1], zorder=1) # Plot the kde curve on the left side of boxplot
                if showBox:
                    for patch in bplot['boxes']: 
                        patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                     patch.set_edgecolor(colors[pos[idx]-1]) # Set boxplot edgecolor to black
                        patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                    for patch in bplot['medians']: 
                        patch.set_color('gold') # Set boxplot median to dark yellow
                        patch.set_linewidth(2) # Set boxplot median line width to 2
            
            except Exception as e: #: continue
                print('Error plotting')
                traceback.print_exc()
                
        plt.xticks(pos, groupsNames)
        plt.xlim(pos[0]-1, pos[-1]+1)
        plt.ylabel(between)