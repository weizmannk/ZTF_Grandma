import os
import glob
import pandas as pd
import numpy as np
from astropy.io import ascii
import optparse
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# =============================================================================
# returns current working directory of a process.
# =============================================================================
datapath = os.getcwd()
# =============================================================================
#  input the csv file
# =============================================================================
csv_file = glob.glob((datapath+'/**/*.csv').split("/")[-1], recursive=True)
save_file_name = [f.split(".")[0] for f in csv_file]

# =============================================================================
# linear regeression  using Max liklihood bayesain  model
# =============================================================================

# =============================================================================
# coeffiscient of determination
# =============================================================================

def coef_determination(y_val, pred):
    u = ((y_val - pred)**2).sum()
    v = ((y_val - y_val.mean())**2).sum()

    return u/v


def sufficient_statistics(t, mag, magerr):
    """[summary]

    :param t: [observation time delays  ]
    :type t: [float in jd time]
    :param mag: [magnitude of observation]
    :type mag: [float]
    :param magerr: [magnitude error]
    :type magerr: [float]
    :return: [ Matrix ]
    :rtype: [type]
    """
    x = t
    y = mag
    sigma = magerr

    # estimate of magerror
    w = 1.0 / sigma**2
    So = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)
    Sxx = np.sum(w * x * x)
    Matrix = np.array([[Sxx, Sx], [Sx, So]])
    vector = np.array([Sxy, Sy])
    return Matrix, vector


# =============================================================================
# Read csv file and  create a base directory (folder) to save the plot
# =============================================================================
def parse_commandline(csv):
    """[Loadind the data of cvs file ]

    :param folder: [output name to save the plots]
    :type folder: [str]
    :param csv: [ intput csv data ]
    :type csv: [type]
    :return: [return a dictionary contain plot directory, cvs file]
    :rtype: [optparse.Values]
    """

    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("-p", "--plotDir", default="outputs")
    parser.add_option("-l", "--lightcurve", default=csv)

    opts, args = parser.parse_args()

    return opts


# ======================================================================
#creat a list to stock essential parametter of data
# ======================================================================
target = []
filter_name = []
evolve_first_seven_days = []
fade_timescale = []
fade_rate = []
evolve_rate = []
time_peak = []
mag_peak = []
error_mag_peak = []


# =============================================================================
# extraction and filters cvs data
# =============================================================================
for i in range(len(csv_file)):
    # Parse command line
    opts = parse_commandline(csv_file[i])
    baseplotDir = opts.plotDir
    if not os.path.isdir(baseplotDir):
        os.makedirs(baseplotDir)
    # =============================================================================
    # plot the lightcurve and Regression models
    # =============================================================================
        
    fig, axs = plt.subplots()
    plt.title(save_file_name[i])
    
    
    # =============================================================================
    # Read csv file
    # =============================================================================
    lc = ascii.read(opts.lightcurve, format='csv')

    # =============================================================================
    # Using filter to create the data table
    # =============================================================================
    for filter in np.unique( lc['filters'] ): #['r', 'R', 'g', 'V', 'I',  'B']:
        tab = lc[(lc['filters'] == filter)]
        
        # =========================================================================
        # select data with more than 2 filters i.e len (tab["filter"]) >2
        # =========================================================================
        if filter == "g":
            color = "darkgreen"
            fmt = "o"
            ls = ":"
            offset = 0

        elif filter == "R":
            color = "darkred"
            fmt = "*"
            offset = 0

        elif filter == "V":
            color = "blue"
            fmt = "x"
            ls = ":"
            offset = 0

        elif filter == "r":
            color = "red"
            fmt = "8"
            ls = ":"
            offset = 0

        elif filter == "B":
            color = "darkblue"
            fmt = "s"
            ls = ":"
            offset = 0

        elif filter == "I":
            color = "yellowgreen"
            fmt = "h"
            ls = ":"
            offset = 0
            
        elif filter == "rmag":
            color = "k"
            fmt = "^"
            ls = ":"
            offset = 0

        elif filter == "gmag":
            color = "olive"
            fmt = "d"
            ls = ":"
            offset = 0
            
        else:
            color = "k"
            fmt = "4"
            ls = ":"
            offset = 0
        
        if filter == 'r'  or 'g':
            if len(tab) > 2:
                tab.rename_column("magerr", "e_mag")
                tab.rename_column("time", "mjd")

                # =====================================================================
                # data selected and reclasse by time increasing0
                # =====================================================================
                tab.sort('mjd')
                idx = np.where(tab['mag'] > 50)[0]
                tab['mag'][idx] = np.nan

                # ======================================================================
                # print and listed, time, magnitude and astronomer
                # ======================================================================
                print("Data that will be used for the fit :")
                print("MJD, mag, instrument")
                for l in tab:
                    print(l['mjd'], l['mag'], l['instrument'])
                    print("---")

                # ======================================================================
                # Treat each file in different way
                # ======================================================================
                f = ["ZTF21abbzjeq", "ZTF21abfmbix", "ZTF21abfaohe", "ZTF21absvlrr", "ZTF21acceboj"]

                if save_file_name[i] in f :
                    # ======================================================================
                    # try to find the index of magnitude  max
                    # ======================================================================
                    indx = np.where(tab['mag'] == np.min(tab['mag']))
                    t_pic = tab['mjd'][indx]

                    if len(t_pic) > 1:
                        t_pic = np.min(t_pic)
                    else:
                        t_pic = t_pic[0]
                    t_peak = np.round(t_pic, 1)
                    y_peak = np.min(tab['mag'])
                    dy_peak = tab['e_mag'][indx]
                    dy_peak = np.round(dy_peak[0], 3)
                    # ======================================================================
                    #the  selection of the seven first days increasing
                    # values from min to magnitude
                    # ======================================================================
                    tmin = np.min(tab[tab['mag'] < 50]['mjd'])-2
                    t_seven_days = 9 + tmin  # First seven days since ZTF detection
                    tab_inc = tab[tab['mjd'] <= t_seven_days]
                    tab_inc.sort('mjd')
                    # ======================================================================
                    # increasing time observation  max and min
                    # ======================================================================
                    tmax_inc = np.min(tab_inc[tab_inc['mag'] < 50]['mjd'])
                    tmin_inc = np.min(tab_inc[tab_inc['mag'] < 50]['mjd'])-2

                    t_inc = tab_inc['mjd']
                    y_inc = tab_inc['mag']
                    dy_inc = tab_inc["e_mag"]
                    #dy_inc = np.sqrt(dy_inc**2 + 0.1**2)  # np.mean

                    # ======================================================================
                    # select the decreasing of fading values from min to magnitude pic and
                    # redefine the names of  time , magnitude and mag-error
                    # ======================================================================
                    tab_dec = tab[tab['mjd'] >= t_pic]
                    tab_dec.sort('mjd')
                    # ======================================================================
                    # time observation  max and min
                    # ======================================================================
                    tmax_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])
                    tmin_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])-2

                    t_dec = tab_dec['mjd']
                    y_dec = tab_dec['mag']
                    dy_dec = tab_dec["e_mag"]
                    #dy_dec = np.sqrt(dy_dec**2 + 0.1**2)  # np.mean

                    # ======================================================================
                    # print and listed, time, magnitude and instrument
                    # ======================================================================
                    print("Data that will be used for the fit :")
                    print("MJD, mag, instrument")
                    for l in tab_inc:
                        print(l['mjd'], l['mag'], l['instrument'])
                        print("--------------------------------------------------")
                    for k in tab_dec:
                        print(k['mjd'], k['mag'], k['instrument'])
                        print("--------------------------------------------------")
                        

                    try:
                        # =============================================================================
                        #  the increase part
                        # sufficient statistics to get the likelihood maximun, bayesian approach
                        # =============================================================================
                        M_inc, v_inc = sufficient_statistics(t_inc, y_inc, dy_inc)
                        x1_inc, x2_inc = np.linalg.solve(M_inc, v_inc)
                        tmax_inc = np.max(t_inc)
                        tt = np.linspace(tmin_inc, tmax_inc, 1000)
                        max_likelihood_inc = x1_inc*t_inc + x2_inc
                        slope_inc = x1_inc

                        # =============================================================================
                        # the decrease likelihood
                        # sufficient statistics to get the likelihood maximun, bayesian approach
                        # =============================================================================
                        M_dec, v_dec = sufficient_statistics(t_dec, y_dec, dy_dec)
                        x1_dec, x2_dec = np.linalg.solve(M_dec, v_dec)
                        tmax_dec = np.max(t_dec)
                        tt = np.linspace(tmin_dec, tmax_dec, 1000)
                        max_likelihood_dec = x1_dec*t_dec + x2_dec
                        slope_dec = x1_dec
                
                        axs.errorbar(t_inc-tmin_inc, y_inc+offset, dy_inc , fmt= fmt, c=color)
                        axs.plot(t_inc - tmin_inc, max_likelihood_inc + offset, c=color, ls=ls, linewidth=1)
                        axs.errorbar(t_dec-tmin_inc, y_dec+offset, dy_dec, fmt=fmt, c=color, label=filter)
                        axs.plot(t_dec - tmin_inc, max_likelihood_dec+offset, c=color, ls=ls,  linewidth=1)
                        
                        
                            
                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            
                            axs.errorbar(t_inc-tmin_inc, y_inc+offset, dy_inc , fmt= fmt, c=color)
                            axs.errorbar(t_dec-tmin_inc, y_dec+offset, dy_dec, fmt=fmt, c=color, label=filter)
                            pass 
                        else:
                            raise
                
                
                elif save_file_name[i] == "ZTF21abultbr":

                    # ======================================================================
                    # try to find the index of magnitude  max
                    # ======================================================================
                    indx = np.where(tab['mag'] == np.min(tab['mag']))
                    t_pic = tab['mjd'][indx]
                    t_peak = np.round(t_pic[0], 1)
                    y_peak= np.min(tab['mag'])
                    dy_peak = tab["e_mag"][indx]
                    dy_peak = np.round(dy_peak[0], 3)
                    
                    tab_dec = tab
                    tab_dec.sort('mjd')
                    # ======================================================================
                    # time observation  max and min
                    # ======================================================================
                    tmax_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])
                    tmin_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])-2

                    t_dec = tab_dec['mjd']
                    y_dec = tab_dec['mag']
                    dy_dec = tab_dec["e_mag"]
                    #dy_dec = np.sqrt(dy_dec**2 + 0.1**2)  # np.mean

                    # ======================================================================
                    # print and listed, time, magnitude and instrument
                    # ======================================================================
                    print("Data that will be used for the fit:")
                    for k in tab_dec:
                        print(k['mjd'], k['mag'], k['instrument'])
                        print("--------------------------------------------------")
                    try:
                        # =============================================================================
                        # the decrease likelihood
                        # sufficient statistics to get the likelihood maximun, bayesian approach
                        # =============================================================================
                        M_dec, v_dec = sufficient_statistics(t_dec, y_dec, dy_dec)
                        x1_dec, x2_dec = np.linalg.solve(M_dec, v_dec)
                        tmax_dec = np.max(t_dec)
                        tt = np.linspace(tmin_dec, tmax_dec, 1000)
                        max_likelihood_dec = x1_dec*t_dec + x2_dec
                        slope_dec = x1_dec

                        axs.errorbar(t_dec-tmin_dec, y_dec, dy_dec, fmt=fmt, c=color, label= filter)
                        axs.plot(t_dec - tmin_dec, max_likelihood_dec, c=color, ls=ls,  linewidth=1)


                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            axs.errorbar(t_dec-tmin_dec, y_dec, dy_dec, fmt=fmt, c=color, label= filter)
                        else:
                            raise
                elif save_file_name[i] == "ZTF21ablssud":
                    
                    # ======================================================================
                    # try to find the index of magnitude  max
                    # ======================================================================
                    indx = np.where(tab['mag'] == np.min(tab['mag']))
                    t_pic = tab['mjd'][indx]
                    t_peak = np.round(t_pic[0], 1)
                    y_peak= np.min(tab['mag'])
                    dy_peak = tab['e_mag'][indx]
                    dy_peak = np.round(dy_peak[0], 3)
                    
                    
                    if filter == "r":
                        tab_dec = tab[tab["mjd"]<=15]
                    else : 
                        tab_dec = tab
                    tab_dec.sort('mjd')
                    # ======================================================================
                    # time observation  max and min
                    # ======================================================================
                    tmax_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])
                    tmin_dec = np.min(tab_dec[tab_dec['mag'] < 50]['mjd'])-2

                    t_dec = tab_dec['mjd']
                    y_dec = tab_dec['mag']
                    dy_dec = tab_dec["e_mag"]
                    #dy_dec = np.sqrt(dy_dec**2 + 0.1**2)  # np.mean

                    # ======================================================================
                    # print and listed, time, magnitude and instrument
                    # ======================================================================
                    print("Data that will be used for the fit:")
                    for k in tab_dec:
                        print(k['mjd'], k['mag'], k['instrument'])
                        print("--------------------------------------------------")
                    try:
                        # =============================================================================
                        # the decrease likelihood
                        # sufficient statistics to get the likelihood maximun, bayesian approach
                        # =============================================================================
                        M_dec, v_dec = sufficient_statistics(t_dec, y_dec, dy_dec)
                        x1_dec, x2_dec = np.linalg.solve(M_dec, v_dec)
                        tmax_dec = np.max(t_dec)
                        tt = np.linspace(tmin_dec, tmax_dec, 1000)
                        max_likelihood_dec = x1_dec*t_dec + x2_dec
                        slope_dec = x1_dec

                        axs.errorbar(t_dec-tmin_dec, y_dec , dy_dec, fmt=fmt, c=color, label= filter)
                        axs.plot(t_dec - tmin_dec, max_likelihood_dec , c=color, ls=ls,  linewidth=1)
                        
                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            axs.errorbar(t_dec-tmin_dec, y_dec , dy_dec, fmt=fmt, c=color, label= filter)
                        else:
                            raise

                
        else :

            tab.rename_column("magerr", "e_mag")
            tab.rename_column("time", "mjd")
            if len(tab) > 1:
                # =====================================================================
                # data selected and reclasse by time increasing0
                # =====================================================================
                tab.sort('mjd')
                idx = np.where(tab['mag'] > 50)[0]
                tab['mag'][idx] = np.nan

                # ======================================================================
                # print and listed, time, magnitude and astronomer
                # ======================================================================
                print("Data that will be used for the fit :")
                print("MJD, mag, instrument")
                for l in tab:
                    print(l['mjd'], l['mag'], l['instrument'])
                    print("---")
                    
                # ======================================================================
                # increasing time observation  max and min
                # ======================================================================
                tmax= np.min(tab[tab['mag'] < 50]['mjd'])
                tmin = np.min(tab[tab['mag'] < 50]['mjd'])-2

                t = tab['mjd']
                y = tab['mag']
                dy = tab["e_mag"]

                try:
                    # =============================================================================
                    #  the increase part
                    # sufficient statistics to get the likelihood maximun, bayesian approach
                    # =============================================================================
                    M, v = sufficient_statistics(t, y, dy)
                    x1, x2 = np.linalg.solve(M, v)
                    tmax = np.max(t)
                    tt = np.linspace(tmin, tmax, 1000)
                    max_likelihood = x1*t + x2
                    slope = x1

                    axs.errorbar(t -tmin , y +offset, dy  , fmt= fmt, c=color, label=filter)
                    axs.plot(t  - tmin , max_likelihood  + offset, c=color, ls=ls, linewidth=1)
                    
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        
                        axs.errorbar(t -tmin , y +offset, dy  , fmt= fmt, c=color,label=filter)
            
                        pass 
                    else:
                        raise
                


    # Create a dictionary to save the slope data of each filter
    # =============================================================================

    axs.set_ylabel(r'Magnitude')
    axs.set_xlabel(r'First time since the ztf detection')

    axs.legend( shadow=True, fancybox=True, loc='best')
    plt.tight_layout()
    axs.grid(True)
    axs.invert_yaxis()
    if not os.path.isdir(baseplotDir):
        os.makedirs(baseplotDir)
        
    plotName = baseplotDir+"/" + save_file_name[i] + ".png"
    plt.savefig(plotName)
    plt.close()
