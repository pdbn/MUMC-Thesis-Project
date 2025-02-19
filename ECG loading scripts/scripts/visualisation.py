# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:38:55 2024

@author: Jip de Kok
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_ecgs_per_location(data, save = False, save_path = None,
                           save_name = "ecgs_per_location",
                           filetype = "svg"):
    '''
    

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame containing the ECG metadata to be plotted.
    save : Bool, optional
        Boolean indicating wether the plot should be saved locally. The default
        is False.
    save_path : Str, optional
        String specifying the path to the directory where the plot should be
        saved. Default is None.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        String indicating what datatype to export (e.g.,  'png', 'svg' or'jpg').
        The default is "svg".

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''
    
    # Extract number of ECGs per location, and only keep those with at least 1000 occurances
    locations = data.LocationName.value_counts()[data.LocationName.value_counts()>1000].reset_index()
    locations.columns = ["Location", "Number_of_ECGs"]
    
    # Initiate figure
    plt.figure()
    
    # Plot number of ECGs per location in a bar chart
    ax = sns.barplot(data = locations, y = "Location", x = "Number_of_ECGs")
    ax.tick_params(axis='x', rotation=90)

    plt.show()
    
    # Save figure locally
    if save:
        plt.savefig(f'{save_path}/{save_name}.{filetype}',
                    bbox_inches='tight')
    
    return ax


def plot_ecgs_at_ICU(data, save = False, save_path = None,
                           save_name = "ecgs_per_location",
                           filetype = "svg"):
    '''
    

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame containing the ECG metadata to be plotted.
    save : Bool, optional
        Boolean indicating wether the plot should be saved locally. The default
        is False.
    save_path : Str, optional
        String specifying the path to the directory where the plot should be
        saved. Default is None.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        String indicating what datatype to export (e.g.,  'png', 'svg' or'jpg').
        The default is "svg".

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''
    
    def count_IC(x):
        counts = x.value_counts()
        
        if(any((np.isin("VERPLEEGAFD. F3", counts.index),
               np.isin("VERPLEEGAFD. H3", counts.index),
               np.isin("VERPLEEGAFD. D3", counts.index),
               np.isin("VERPLEEGAFD. E3", counts.index)))):
            return("At least one ECG at ICU")
        else:
            return("No ECG at ICU")
        

    ECG_at_ICU = data.groupby("PatientID").LocationName.apply(
        count_IC).value_counts().reset_index()
    
    ECG_at_ICU.columns = ["Location", "Number_of_patients"]


    plt.figure()

    ax = sns.barplot(data = ECG_at_ICU, y = "Number_of_patients", x = "Location")
    plt.show()
    
    # Save figure locally
    if save:
        plt.savefig(f'{save_path}/{save_name}.{filetype}',
                    bbox_inches='tight')
    
    return ax


def plot_ecgs_per_year(data, save = False, save_path = None,
                           save_name = "ecgs_per_location",
                           filetype = "svg"):
    '''
    

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame containing the ECG metadata to be plotted.
    save : Bool, optional
        Boolean indicating wether the plot should be saved locally. The default
        is False.
    save_path : Str, optional
        String specifying the path to the directory where the plot should be
        saved. Default is None.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        String indicating what datatype to export (e.g.,  'png', 'svg' or'jpg').
        The default is "svg".

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    '''
    data.loc[:,"year"] = data.AcquisitionDateTime.dt.year
    
    ecg_per_year = data.groupby("year").nunique().reset_index()
    
    plt.figure()
    
    ax = sns.lineplot(data = ecg_per_year, x = "year", y = "PatientID",
                      label = "unique patients")
    ax = sns.lineplot(data = ecg_per_year, x = "year", y = "filename",
                      label = "total ECGS")
    ax.legend()

    plt.show()
    
    # Save figure locally
    if save:
        plt.savefig(f'{save_path}/{save_name}.{filetype}',
                    bbox_inches='tight')