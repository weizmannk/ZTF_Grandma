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

target = []
filter_name = []
evolve_first_seven_days = []
fade_timescale = []
fade_rate = []
evolve_rate = []
time_peak =[]
mag_peak =[]

filt = "gr"

# =============================================================================
# Using filter to create the data table
# =============================================================================
#for filter in lc['filters']:

# ======================================================================
# create new folder by each filter to put each filter plot in
# ======================================================================


# =============================================================================
# plot the lightcurve and Regression models
# =============================================================================

fig, axs = plt.subplots()
fig.suptitle("ZTF21.....")
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
    # Read csv file
    # =============================================================================
    lc = ascii.read(opts.lightcurve, format='csv')

    if save_file_name[i] == "ZTF21abbzjeq":
        r_color = "darkred"
        g_color = "darkgreen"
        r_fmt  =   "d"
        g_fmt  =   "o"
    elif save_file_name[i] == "ZTF21absvlrr":
        r_color = "lightsalmon"
        g_color = "green"
        r_fmt  =   "^"
        g_fmt  =   "h"
    elif save_file_name[i] == "ZTF21abfmbix":
        r_color = "darkblue"
        g_color = "blue"
        r_fmt   = "s"
        g_fmt   = "+"
    elif save_file_name[i] == "ZTF21abfaohe":
        r_color = "red"
        g_color = "blue"
        r_fmt   =   "8"
        g_fmt  =     "*"

    for filter in ["g", "r"] :
        tab = lc[(lc['filters'] == filter)]
        
        offset = 0

        # =========================================================================
        # select data with more than 2 filters i.e len (tab["filter"]) >2
        # =========================================================================
        if len(tab) > 3:
            tab.rename_column("magerr", "e_mag")
            tab.rename_column("time", "mjd")
            
            # =====================================================================
            # data selected and reclasse by time increasing0
            # =====================================================================
            tab.sort('mjd')
            idx = np.where(tab['mag'] > 50)[0]
            tab['mag'][idx] = np.nan

            # ======================================================================
            # print and listed, time, magnitude and instrument
            # ======================================================================
            print("Data that will be used for the fit:")
            print("MJD, mag, instrument")
            for l in tab:
                print(l['mjd'], l['mag'], l['instrument'])
                print("---")

            # ======================================================================
            # try to find the index of magnitude  max
            # ======================================================================
            indx = np.where(tab['mag'] == np.min(tab['mag']))
            t_pic = tab['mjd'][indx]
            t_peak = np.round(t_pic[0], 1)
            y_peak= np.min(tab['mag'])
            # ======================================================================
            #the  selection of the seven first days increasing
            # values from min to magnitude
            # ======================================================================
            tmin = np.min(tab[tab['mag'] < 50]['mjd'])-2
            t_seven_days = 9 + tmin    #  7 days in tim
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
            dy_inc = np.sqrt(dy_inc**2 + 0.1**2)  # np.mean

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
            dy_dec = np.sqrt(dy_dec**2 + 0.1**2)  # np.mean

            # ======================================================================
            # print and listed, time, magnitude and instrument
            # ======================================================================
            print("Data that will be used for the fit:")
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
                
                if filter == "r":
                
                    axs.errorbar(t_inc-tmin_inc, y_inc+offset, dy_inc , fmt= g_fmt, c=g_color, label= filter)
                    axs.plot(t_inc - tmin_inc, max_likelihood_inc + offset, 'k--', linewidth=1)
                    axs.errorbar(t_dec-tmin_inc, y_dec +offset, dy_dec, fmt=g_fmt, c=g_color)
                    axs.plot(t_dec - tmin_inc, max_likelihood_dec+offset, 'k-', linewidth=1)

                    indy_inc= np.where(y_inc==np.min(y_inc))
                    x_inc = t_inc[indy_inc]
                    y_inc = np.min(y_inc) #+offset,  # np.min(max_likelihood_inc)

                    indy_dec = np.where(y_dec == np.max(y_dec))
                    x_dec = t_dec[indy_dec]
                    y_dec = np.max(y_dec) #  #np.max(max_likelihood_dec)
        
                    #axs.text(x_inc, y_inc+offset, r'   %.2f ' % slope_inc, rotation=90, color="darkgreen", fontsize= 12)
                
                else :
                    axs.errorbar(t_inc-tmin_inc, y_inc, dy_inc, fmt=r_fmt, c=r_color, label=filter)
                    axs.plot(t_inc - tmin_inc, max_likelihood_inc, 'k--', linewidth=1 )
                    axs.errorbar(t_dec-tmin_inc, y_dec , dy_dec, fmt=r_fmt, c=r_color)
                    axs.plot(t_dec - tmin_inc, max_likelihood_dec, 'k-', linewidth=1)

                    indy_inc = np.where(y_inc == np.max(y_inc))
                    x_inc = t_inc[indy_inc]
                    y_inc = np.max(y_inc)

                    indy_dec = np.where(y_dec == np.max(y_dec))
                    x_dec = t_dec[indy_dec]
                    y_dec = np.max(y_dec)  # np.max(max_likelihood_dec)
    
                    #axs.text(x_inc, y_inc, r'  %.2f ' % slope_inc,  rotation= 90, color="darkred", fontsize= 12)
                    #axs.text(x_dec, y_dec, r'  %.2f  ' % slope_dec,  rotation=90,  color="darkred", fontsize= 12)
        
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    pass
                else:
                    raise

        # =============================================================================
        # Stock all the essential data about fade/evolve rate, timescale.... 
        # =============================================================================
            target.append(save_file_name[i])
            filter_name.append(filter)
            evolve_first_seven_days.append(np.max(t_inc) - np.min(t_inc))
            fade_timescale.append(np.max(t_dec) - np.min(t_dec))
            evolve_rate.append(slope_inc)
            fade_rate.append(slope_dec)
            time_peak.append(t_peak)
            mag_peak.append(y_peak)

        else:
            print("***********************************************************")
            print("There are not enough points")
            print("===========================================================")
    # =============================================================================
    # Create a dictionary to save the slope data of each filter
    # =============================================================================

    dic = {"target": target,
        "filter": filter_name,
        "Peak_time": time_peak,
        "Peak_Mag": mag_peak,
        "evolve_timescale": evolve_first_seven_days,
        "evolve_rate": evolve_rate,
        "fade_timescale": fade_timescale,
        "fade_rate": fade_rate}

  
#axs.margins(y=0.7)
axs.set_ylabel(r'Magnitude')
axs.set_xlabel(r'First time since the ztf detection')

axs.grid(True)
axs.legend(shadow=True,  ncol=1, fancybox=True, loc='best')
axs.invert_yaxis()


plotName = "plot.pdf"
plt.savefig(plotName)
plt.close()

df = pd.DataFrame(dic)

# ======================================================================
# create new folder to save data
# ======================================================================
saveDir = os.path.join(baseplotDir, "slope")
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

# ======================================================================
# Create a csv file of essential data
# ======================================================================
df.to_csv(saveDir+"/"+"slope.csv")
