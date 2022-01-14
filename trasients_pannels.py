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
# plot the lightcurve and Regression models
# =============================================================================


fig, (axs1, axs2) = plt.subplots(2, 1)
# "abbzjeq;" + " abfmbix;"+" abfaohe;"+" absvlrr;" + " ablssud;"+ " abultbr;"+" acceboj")

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
    
    # =============================================================================
    # Using filter to create the data table
    # =============================================================================   
    for filter in ["g", "r"]:
        tab = lc[(lc['filters'] == filter)]

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
            print("Data that will be used for the fit :")
            print("MJD, mag, instrument")
            for l in tab:
                print(l['mjd'], l['mag'], l['instrument'])
                print("---")
            
            # ======================================================================
            # Treat each file in different way 
            # ======================================================================
            f =["ZTF21abbzjeq", "ZTF21abfmbix", "ZTF21abfaohe", "ZTF21absvlrr", "ZTF21acceboj"]
            
            if save_file_name[i] in f:   
                # ======================================================================
                # try to find the index of magnitude  max
                # ======================================================================
                indx = np.where(tab['mag'] == np.min(tab['mag']))
                t_pic = tab['mjd'][indx]
                t_peak = np.round(t_pic[0], 1)
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
                #dy_dec = np.sqrt(dy_dec**2 + 0.1**2)  # np.meandian

                    
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
                    
                    if save_file_name[i] == "ZTF21abbzjeq":
                        r_color = "darkred"
                        g_color = "darkgreen"
                        r_fmt  =   "d"
                        g_fmt  =   "o"
                        col_inc = "b"
                        ls_inc = ":"
                        col_dec = "b"
                        ls_dec = "-"
                        
                    elif save_file_name[i] == "ZTF21absvlrr":
                        r_color = "lightsalmon"
                        g_color = "yellowgreen"
                        r_fmt  =   "^"
                        g_fmt  =   "h"
                        col_inc = "r"
                        ls_inc = ":"
                        col_dec = "r"
                        ls_dec = "-"
                            
                    elif save_file_name[i] == "ZTF21abfmbix":
                        r_color = "brown"
                        g_color = "teal"
                        r_fmt   = "s"
                        g_fmt   = "4"
                        col_inc = "k"
                        ls_inc = ":"
                        col_dec = "k"
                        ls_dec = "-"
                        
                    elif save_file_name[i] == "ZTF21abfaohe":
                        r_color = "red"
                        g_color = "seagreen"
                        r_fmt   =   "8"
                        g_fmt  =     "*"
                        col_inc = "b"
                        ls_inc = ":"
                        col_dec = "k"
                        ls_dec = ":"
                        
                    elif save_file_name[i] == "ZTF21acceboj":
                        r_color = "yellow"
                        g_color = "limegreen"
                        r_fmt   =   "8"
                        g_fmt  =     "*"
                        col_inc = "b"
                        ls_inc = ":"
                        col_dec = "k"
                        ls_dec = ":"
                    
                    else:
                        pass
                    
                    if filter == "g":

                        #axs.errorbar(t_inc-tmin_inc, y_inc, dy_inc , fmt= g_fmt, c=g_color, label= filter)
                        #axs.plot(t_inc - tmin_inc, max_likelihood_inc + offset, c=col_inc, ls=ls_inc, linewidth=1)
                        axs1.errorbar(t_dec-tmin_inc, y_dec, dy_dec, fmt=g_fmt, c=g_color, label= save_file_name[i])
                        axs1.plot(t_dec - tmin_inc, max_likelihood_dec, c=col_dec, ls=ls_dec,  linewidth=1)

                        if save_file_name[i] == "ZTF21abbzjeq":                          
                            indy_dec = np.where(y_dec == np.max(y_dec))
                            x_dec = t_dec[indy_dec]-15
                            y_dec = np.max(y_dec)-0.4 
                            axs1.text(x_dec, y_dec, r'   %.3f ' % slope_dec, rotation=-20, color="navy", fontsize=10)
                        
                        elif save_file_name[i] == "ZTF21absvlrr":
                            indy_dec = np.where(y_dec == np.max(y_dec))
                            x_dec = t_dec[indy_dec]-5
                            y_dec = np.max(y_dec)-0.1 
                            axs1.text(x_dec, y_dec, r'  %.3f ' % slope_dec, rotation=-20, color="navy", fontsize=10)


                        elif save_file_name[i] == "ZTF21abfmbix":                    
                            indy_dec = np.where(y_dec == np.max(y_dec))
                            x_dec = t_dec[indy_dec]-5.5
                            y_dec = np.max(y_dec) -0.2
                            axs1.text(x_dec, y_dec, r'  %.3f ' % slope_dec, rotation=-20, color="navy", fontsize=10)

                        elif save_file_name[i] == "ZTF21abfaohe":  
                            indy_dec = np.where(y_dec == np.max(y_dec))
                            x_dec = t_dec[indy_dec]-20
                            y_dec = np.max(y_dec)-1.5
                            axs1.text(x_dec, y_dec, r'  %.3f ' % slope_dec,rotation=-13, color="navy", fontsize=10)
                        
                        elif save_file_name[i] == "ZTF21acceboj":
                            indy_dec = np.where(y_dec == np.max(y_dec))
                            x_dec = t_dec[indy_dec]-10
                            y_dec = np.max(y_dec)-0.2
                            axs1.text(x_dec, y_dec, r'  %.3f ' % slope_dec,  rotation=0, color="navy", fontsize=10)

                        else:
                            pass
   
                    else :
                        if save_file_name[i] == "ZTF21acceboj":
                            pass
                        
                        else:

                            #axs.errorbar(t_inc-tmin_inc, y_inc, dy_inc, fmt=r_fmt, c=r_color, label=filter)
                            #axs.plot(t_inc - tmin_inc, max_likelihood_inc, c=col_inc, ls=ls_inc,  linewidth=1 )
                            axs2.errorbar(t_dec-tmin_inc, y_dec, dy_dec, fmt=r_fmt, c=r_color, label= save_file_name[i])
                            axs2.plot(t_dec - tmin_inc, max_likelihood_dec, c=col_dec, ls=ls_dec,  linewidth=1)
                            
                            if save_file_name[i] == "ZTF21abbzjeq":                          
                                indy_dec = np.where(y_dec == np.max(y_dec))
                                x_dec = t_dec[indy_dec]-23
                                y_dec = np.max(y_dec)-0.6 
                                axs2.text(x_dec, y_dec, r'   %.3f ' % slope_dec, rotation=-15, color="navy", fontsize=10)
                            
                            elif save_file_name[i] == "ZTF21absvlrr":
                                indy_dec = np.where(y_dec == np.max(y_dec))
                                x_dec = t_dec[indy_dec]-10
                                y_dec = np.max(y_dec) -0.15
                                axs2.text(x_dec, y_dec, r'   %.3f ' % slope_dec, rotation=-10, color="navy", fontsize=10)


                            elif save_file_name[i] == "ZTF21abfmbix":                    
                                indy_dec = np.where(y_dec == np.max(y_dec))
                                x_dec = t_dec[indy_dec]-15
                                y_dec = np.max(y_dec)+0.5 
                                axs2.text(x_dec, y_dec, r'   %.3f ' % slope_dec, rotation=25, color="navy", fontsize=10)
                            
                            elif save_file_name[i] == "ZTF21abfaohe":
                                indy_dec = np.where(y_dec == np.max(y_dec))
                                x_dec = t_dec[indy_dec]-25
                                y_dec = np.max(y_dec)-0.5
                                axs2.text(x_dec, y_dec, r'   %.3f ' % slope_dec, rotation=-15, color="navy", fontsize=10)

                            else : 
                                pass

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
                error_mag_peak.append(dy_peak)
                
            elif save_file_name[i] == "ZTF21ablssud":
    
                r_color = "red"
                g_color = "darkolivegreen"
                r_fmt  =   "x"
                g_fmt  =   "h"
                    
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
                    
                    if filter == "g":
                       
        
                        axs1.errorbar(t_dec-tmin_dec, y_dec , dy_dec, fmt=g_fmt, c=g_color, label=save_file_name[i])
                        axs1.plot(t_dec - tmin_dec, max_likelihood_dec , 'k:', linewidth=1)
                        indy_dec = np.where(y_dec == np.max(y_dec))
                        x_dec = t_dec[indy_dec]-4
                        y_dec = np.max(y_dec)-0.05#  #np.max(max_likelihood_dec)
                        axs1.text(x_dec, y_dec, r'  %.3f  ' % slope_dec,  rotation=-20,  color="navy", fontsize=10)
                            
            
                    else :
                        axs2.errorbar(t_dec-tmin_dec, y_dec, dy_dec, fmt=r_fmt, c=r_color, label=save_file_name[i])
                        axs2.plot(t_dec - tmin_dec, max_likelihood_dec, 'k:', linewidth=1)

                        indy_dec = np.where(y_dec == np.max(y_dec))
                        x_dec = t_dec[indy_dec]-10
                        y_dec = np.max(y_dec)-0.5  # np.max(max_likelihood_dec)
        
                        axs2.text(x_dec, y_dec, r'  %.3f  ' % slope_dec,rotation=-45,  color="navy", fontsize=10)
                        
    
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        pass
                    else:
                        raise

            elif save_file_name[i] == "ZTF21abultbr":
                r_color = "darkorange"
                g_color = "seagreen"
                r_fmt   = "H"
                g_fmt   = "s"

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


                    axs2.errorbar(t_dec-tmin_dec, y_dec, dy_dec, fmt=r_fmt, c=r_color, label= save_file_name[i] )
                    axs2.plot(t_dec - tmin_dec, max_likelihood_dec, 'g:', linewidth=1)

                    indy_dec = np.where(y_dec == np.max(y_dec))
                    x_dec = t_dec[indy_dec]+1.5
                    y_dec = np.max(y_dec) - 0.1  # np.max(max_likelihood_dec)

                    axs2.text(x_dec, y_dec, r'  %.3f  ' % slope_dec,  rotation=0,  color="navy", fontsize=10)
                    
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        pass
                    else:
                        raise
    
# =============================================================================
# Create a dictionary to save the slope data of each filter
# =============================================================================

dic_f = {"target": target,
        "filter": filter_name,
        "Peak_time": time_peak,
        "Peak_Mag": mag_peak,
        "Peak_Mag_eror": error_mag_peak,
        "evolve_timescale": evolve_first_seven_days,
        "evolve_rate": evolve_rate,
        "fade_timescale": fade_timescale,
        "fade_rate": fade_rate}



df_f = pd.DataFrame(dic_f)
# ======================================================================
# create new folder to save data
# ======================================================================
saveDir = os.path.join(baseplotDir, "slope")
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
# ======================================================================
# Create a csv file of essential data
# ======================================================================
df_f.to_csv(saveDir+"/"+"slopeFPannels.csv")
                

#axs.margins(y=0.7)

axs1.text(45, 16.5, "g'-band ",   rotation=-45, color="k", fontsize=14)
axs2.text(70, 17, "r'-band ",   rotation=-45, color="k", fontsize=14)
axs1.set_ylabel(r'Magnitude')

axs2.set_ylabel(r'Magnitude')
axs2.set_xlabel(r'First time since the peak mag detection')


axs1.legend(bbox_to_anchor=(1, 1), shadow=True, fancybox=True, loc='best')

axs2.legend(bbox_to_anchor=(1, 1),  shadow=True, fancybox=True, loc='best')

axs1.grid(True)
axs1.invert_yaxis()

axs2.grid(True)
axs2.invert_yaxis()

plt.tight_layout()



plotName = "Trasients_pannels.png"
plt.savefig(plotName)
plt.close()
