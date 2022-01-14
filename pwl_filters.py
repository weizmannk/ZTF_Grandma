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
# score
# =============================================================================
def coef_determination(y_val, pred):
    u = ((y_val - pred)**2).sum()
    v = ((y_val - y_val.mean())**2).sum()

    return u/v

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

# =============================================================================
# Regeression using the pymultinest (bayesain model)
# =============================================================================

def plaw(t, f0, alpha=1.0, t0=None):
    """
    Power law function:
    f / f0 = (t - t0) ** -alpha
    """
    return f0 * (t - t0) ** -alpha

def pmag(t, f0, alpha=1.0, t0=None):
    """
    Use indices from plaw fit to calculate magnitudes
    -2.5 log(f/f0) = 2.5 * alpha * log(dt)
    m = -2.5 log(f)
    m + 2.5 log(f0) = 2.5 * alpha * log(dt)
    """
    return 2.5 * alpha * np.log10(t - t0) - 2.5 * np.log10(f0)

def eflux(emag, flux):
    """
    Error propoagation:
    m - m0 = dm = 2.5 log(1 + df / f0)
    10.0 ** (dm / 2.5) = 1 + df / f0
    df = f0 * (10.0 ** (dm / 2.5) - 1)
    """
    return flux * (10.0 ** (emag / 2.5) - 1)

def myprior(cube, ndim, nparams):
    cube[0] = cube[0]*(tmax - tmin) + tmin
    cube[1] = cube[1]*3.0
    cube[2] = cube[2]*20.0 - 10.0

def myloglike(cube, ndim, nparams):
    t0fit = cube[0]
    alpha = cube[1]
    f0 = 10**cube[2]
    
    mod = pmag(t, f0, alpha=alpha, t0=t0fit)

    idx1 = np.where(~np.isnan(y))[0]
    idx2 = np.where(np.isnan(y))[0]

    chisq = -(1/2)*np.sum(((y[idx1] - mod[idx1])**2.)/(dy[idx1]**2.))/(len(t[idx1]) - nparams)
    return chisq


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
    parameters = ["t0", "alpha", "f0"]
    n_params = len(parameters)

    n_live_points = 1000
    evidence_tolerance = 0.1
    max_iter = -1

    # =============================================================================
    # Using filter to create the data table
    # =============================================================================
    for filter in lc['filters']:
        tab = lc[(lc['filters'] == filter)]

        if filter == "R":
            color = "red"
        elif filter == "B":
            color = "darkblue"
        elif filter == "gmag":
            color = "lightgreen"
        elif filter == "rmag":
            color = "lightsalmon"
        elif filter == "I":
            color = "orange"
        elif filter == "V":
            color = "green"
        elif filter == "g":
            color = "darkgreen"
        elif filter == "r":
            color = "darkred"
        else:
            color = "black"
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
            # create new folder by each filter to put each filter plot in
            # ======================================================================
            plotDir = os.path.join(
                baseplotDir, save_file_name[i]+"/"+filter)
            if not os.path.isdir(plotDir):
                os.makedirs(plotDir)

            # ======================================================================
            # try to find the index of magnitude  max 
            # ======================================================================
            indx = np.where(tab['mag']==np.min(tab['mag']))
            t_pic = tab['mjd'][indx]
    
            # ======================================================================
            #the  selection of the seven first days increasing 
            # values from min to magnitude 
            # ======================================================================
            tmin = np.min(tab[tab['mag'] < 50]['mjd'])-2
            t_seven_days = 7+ tmin    # we use 9  beacause we substructed the 2 days in tim
            tab_inc = tab[tab['mjd'] <= t_seven_days]  
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
                max_likelihood_inc = x1_inc*t_inc+ x2_inc
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

                # ======================================================================
                # create new folder by each filter to put each filter plot in
                # ======================================================================
                plotDir = os.path.join(
                    baseplotDir, save_file_name[i]+"/"+filter)
                if not os.path.isdir(plotDir):
                    os.makedirs(plotDir)
        
                # =============================================================================
                # plot the lightcurve and Regression models
                # =============================================================================
                plotName = plotDir + "/"+filter + ".pdf"

                fig, axs = plt.subplots(2) 
                fig.suptitle(save_file_name[i])
                
                axs[0].errorbar(t_inc-tmin_inc, y_inc, dy_inc,  fmt='o', c= color, label=filter)
                axs[0].plot(t_inc - tmin_inc, max_likelihood_inc, 'k-', linewidth=1, label=r'slope = %.2f '% slope_inc)
                axs[0].set_ylabel(r'Magnitude')
                axs[0].set_xlabel(r'First time since ztf detection to the mag pick t0= % .5f ' % tmin_inc )
                axs[0].grid(True)
                axs[0].legend(loc='best')
                axs[0].invert_yaxis()
            
                axs[1].errorbar(t_dec-tmin_inc, y_dec, dy_dec,  fmt='o', c= color, label=filter)
                axs[1].plot(t_dec - tmin_inc, max_likelihood_dec, 'k-', linewidth=1, label=r'slope = %.2f' % slope_dec)
                axs[1].set_ylabel(r'Magnitude')
                axs[1].set_xlabel(r'First time since the detection pick mag  t0= % .5f ' % tmin_dec )
                axs[1].grid(True)
                axs[1].legend(loc='best')
                axs[1].invert_yaxis()
                
                plt.savefig(plotName)
                plt.close()

            
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    pass
                else:
                    raise
                
        else:
            print("***********************************************************")
            print("There are not enough points")
            print("===========================================================")

