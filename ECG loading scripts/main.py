# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:24:08 2024

@author: Jip de Kok
"""

# Load packages
import sys
import pandas as pd

# Load local scripts
# Add scripts folder to sys.path so scripts can be imported
sys.path.insert(1, 'scripts')
from scripts.data_functions import load_ecg_data,addDamicAdmissionData
from scripts.visualisation import plot_ecgs_per_location, plot_ecgs_at_ICU, plot_ecgs_per_year

# ============= #
# Load ECG data #
# ============= #

# Set quickload to True if you only want to quickly read the ECG metadata.
# Set quickload to False if you want to load all ECG data including the
# waveform data (this can take multiple hours).
quickload = True
makePlots = True # Set to True for overview plots
save = False

# Specify folder directories
data_path = r"L:/SPEC/ICU/RESEARCH/Data-onderzoek/studenten/Econometrie/Bao Phung/ECG loading scripts/data"
local_ecg_path = r"L:/SPEC/ICU/RESEARCH/Data-onderzoek/ECG dataset/XML subset"
figure_path = "figures"

if quickload:
    df_ecg = load_ecg_data(data_path = data_path,
                           quickload = quickload)
else:
    load_ecg_data(local_ecg_path = local_ecg_path, quickload = quickload)

# =============== #
# Add DAM-IC data #
# =============== #
# Load DAM-IC data
mostRecentDamicFile = r"L:\SPEC\ICU\RESEARCH\Data-onderzoek\Basisdataset\DAM-IC\Definitief cohort versies\ADAMICPatientCohort_HrCorrected2013-2023_hashed_2024-10-15.csv"
df_ecg=addDamicAdmissionData(df_ecg, mostRecentDamicFile)

if save:
    df_ecg.to_csv(r"L:\SPEC\ICU\RESEARCH\Data-onderzoek\studenten\Econometrie\Bao Phung\ECG loading scripts\data\DAMIC_ECG_metadata_xx-xx-2025.csv")

# ============== #
# Visualise data #
# ============== #
if makePlots:
    plot_ecgs_per_location(df_ecg, save = save, save_path = figure_path)
    
    plot_ecgs_at_ICU(df_ecg, save = save, save_path = figure_path)
    
    plot_ecgs_per_year(df_ecg, save = save, save_path = figure_path)
