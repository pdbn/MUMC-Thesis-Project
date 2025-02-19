# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:15:01 2024

@author: G10062682
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from ECGXMLReader import ECGXMLReader
from ECG_preprocessing import preprocess_ecg
import time
import os
import re

def load_ecg_data(local_ecg_path: object = None, data_path: object = None, quickload: object = True,
                  preprocess: object = True, scale: object = False, save: object = False) -> DataFrame:
    '''
    This functions loads and prepares ECG data from .XML files into a usable
    format.

    Parameters
    ----------
    local_ecg_path : Str
        String indicating the path to the directory of the ECG .XML files.
    data_path : Str
        String indicating the path to the directory of the ECG metadata.
    quickload : Bool, optional
        Boolean indicating whether only the metadata should be loaded, or if 
        all ECG data including waveform data should be loaded. Set quickload to
        True if you only want to quickly read the ECG metadata that was
        previously generated.Set quickload to False if you want to load all ECG
        data including the waveform data (this can take multiple hours). If this
        is your first time running load_ecg_data, and no metadata has been
        extracted and saved before, then set quickload to False, and save to
        True.
    preprocess : Bool, optional
        Boolean indicating whether waveform data should be pre-processed.
        The default is True.
    scale : Bool, optional
        Boolean indicating wether waveform data should be scaled (standardised).
        The default is False.
    save : Bool, optional
        Boolean indicating wether the ECG metadata should be saved locally. The
        default is False.

    Returns
    -------
    df_ecg : pd.DataFrame
        Pandas DataFrame containing the ECG metadata.
    x_ecg : np.matrix
        Multidimensional numpy matrix containing all waveform data. This data
        will be pre-processed and scaled based on the input parameters. Only 
        returned if quickload=True
    ecg_list : list
        List containing each individual ECG containing raw waveforms as well as 
        metadata. Only returned if quickload=True

    '''
    
    if quickload:
        df_ecg = pd.read_csv(f"{data_path}/ECG_metadata.csv",
                             dtype = {"filename"            : str,
                                      "PatientID"           : np.float64,
                                      "PatientAge"          : np.float64,
                                      "Gender"              : 'category',
                                      "DataType"            : 'category',
                                      "Site"                : np.int64,
                                      "SiteName"            : 'category',
                                      "RoomID"              : 'category',
                                      "AcquisitionDevice"   : 'category',
                                      "Status"              : 'category',
                                      "Location"            : 'category',
                                      "LocationName"        : str,
                                      "VentricularRate"     : np.float64,
                                      "AtrialRate"          : np.float64,
                                      "PRInterval"          : np.float64,
                                      "QRSDuration"         : np.float64,
                                      "QTInterval"          : np.float64,
                                      "QTCorrected"         : np.float64,
                                      "PAxis"               : np.float64,
                                      "RAxis"               : np.float64,
                                      "TAxis"               : np.float64,
                                      "QRSCount"            : np.float64,
                                      "ECGSampleBase"       : np.float64,
                                      "ECGSampleExponent"   : np.float64,
                                      "QTcFrederica"        : np.float64,
                                      "AcquisitionDateTime" : str})
        
        df_ecg.AcquisitionDateTime = pd.to_datetime(df_ecg.AcquisitionDateTime)
    
        return(df_ecg)      
    
    else:
        
        # =========================================================================== #
        #
        #                        Load ECG waveform data
        #
        # =========================================================================== #
        # Read one example ECG file
        ecg = ECGXMLReader(f'{local_ecg_path}/MUSE_20211104_135958_73000.xml',
                           augmentLeads=True)
        
        # Initiate list to store ECGs
        ecg_list = {}
        file_list = os.listdir(local_ecg_path)
        
        # Loop over files and load ECG data iteratively
        t = time.time()
        count = 0
        message_interval = 0.1
        for filename in file_list:
            if not filename.endswith('.XML') or filename.endswith('.xml'): continue
            fullname = os.local_ecg_path.join(local_ecg_path, filename)
            ecg = ECGXMLReader(fullname, augmentLeads = True)
            ecg_list[filename] = ecg
            count += 1
            if count/len(file_list) > message_interval:
                print(f"Finnished {int(100*message_interval)}% ~ "
                      f"{int((time.time() - t)/60)} minutes runtime")
                message_interval += 0.1
        elapsed = time.time() - t
        
        print(f"Loading ECGS took {elapsed/60/60} hours.")
        
        # Initiate df_ecg which will store metadata for all ECGs
        df_ecg = pd.DataFrame(np.nan,
                               index = np.arange(295813),
                               columns = [
                                   "filename",
                                   "PatientID",
                                   "PatientAge",
                                   "Gender",
                                   'DataType',
                                    'Site',
                                    'SiteName',
                                    'RoomID',
                                    'AcquisitionDevice',
                                    'Status',
                                    'Location',
                                    'LocationName',
                                    'AcquisitionTime',
                                    'AcquisitionDate',
                                    'VentricularRate',
                                     'AtrialRate',
                                     'PRInterval',
                                     'QRSDuration',
                                     'QTInterval',
                                     'QTCorrected',
                                     'PAxis',
                                     'RAxis',
                                     'TAxis',
                                     'QRSCount',
                                     'ECGSampleBase',
                                     'ECGSampleExponent',
                                     'QTcFrederica'
                                   ])
    
        # Fill df_ecg with the available metadata
        for pos, element in enumerate(ecg_list):
            try:
                df_ecg.loc[pos, 'filename'] = ecg_list[element].path.split("\\")[-1]
            except:
                df_ecg.loc[pos, 'filename'] = np.nan
            
            try:
                df_ecg.loc[pos, 'PatientID'] = ecg_list[element].PatientDemographics["PatientID"]
            except:
                df_ecg.loc[pos, 'PatientID'] = np.nan
                
            try:
                df_ecg.loc[pos, 'PatientAge'] = ecg_list[element].PatientDemographics["PatientAge"]
            except:
                df_ecg.loc[pos, 'PatientAge'] = np.nan
                
            try:
                df_ecg.loc[pos, 'Gender'] = ecg_list[element].PatientDemographics["Gender"]
            except:
                df_ecg.loc[pos, 'Gender'] = np.nan
                
            try:
                df_ecg.loc[pos, 'VentricularRate'] = ecg_list[element].RestingECGMeasurements["VentricularRate"]
            except:
                df_ecg.loc[pos, 'VentricularRate'] = np.nan
                
            try:
                df_ecg.loc[pos, 'AtrialRate'] = ecg_list[element].RestingECGMeasurements["AtrialRate"]
            except:
                df_ecg.loc[pos, 'AtrialRate'] = np.nan
                
            try:
                df_ecg.loc[pos, 'PRInterval'] = ecg_list[element].RestingECGMeasurements["PRInterval"]
            except:
                df_ecg.loc[pos, 'PRInterval'] = np.nan
                
            try:
                df_ecg.loc[pos, 'QRSDuration'] = ecg_list[element].RestingECGMeasurements["QRSDuration"]
            except:
                df_ecg.loc[pos, 'QRSDuration'] = np.nan
                
            try:
                df_ecg.loc[pos, 'QTInterval'] = ecg_list[element].RestingECGMeasurements["QTInterval"]
            except:
                df_ecg.loc[pos, 'QTInterval'] = np.nan
                
            try:
                df_ecg.loc[pos, 'QTCorrected'] = ecg_list[element].RestingECGMeasurements["QTCorrected"]
            except:
                df_ecg.loc[pos, 'QTCorrected'] = np.nan
                
            try:
                df_ecg.loc[pos, 'PAxis'] = ecg_list[element].RestingECGMeasurements["PAxis"]
            except:
                df_ecg.loc[pos, 'PAxis'] = np.nan
                
            try:
                df_ecg.loc[pos, 'RAxis'] = ecg_list[element].RestingECGMeasurements["RAxis"]
            except:
                df_ecg.loc[pos, 'RAxis'] = np.nan
                
            try:
                df_ecg.loc[pos, 'TAxis'] = ecg_list[element].RestingECGMeasurements["TAxis"]
            except:
                df_ecg.loc[pos, 'TAxis'] = np.nan
                
            try:
                df_ecg.loc[pos, 'QRSCount'] = ecg_list[element].RestingECGMeasurements["QRSCount"]
            except:
                df_ecg.loc[pos, 'QRSCount'] = np.nan
                
            try:
                df_ecg.loc[pos, 'ECGSampleBase'] = ecg_list[element].RestingECGMeasurements["ECGSampleBase"]
            except:
                df_ecg.loc[pos, 'ECGSampleBase'] = np.nan
                
            try:
                df_ecg.loc[pos, 'ECGSampleExponent'] = ecg_list[element].RestingECGMeasurements["ECGSampleExponent"]
            except:
                df_ecg.loc[pos, 'ECGSampleExponent'] = np.nan
                
            try:
                df_ecg.loc[pos, 'QTcFrederica'] = ecg_list[element].RestingECGMeasurements["QTcFrederica"]
            except:
                df_ecg.loc[pos, 'QTcFrederica'] = np.nan
            
            try:
                df_ecg.loc[pos, 'DataType'] = ecg_list[element].TestDemographics["DataType"]
            except:
                df_ecg.loc[pos, 'DataType'] = np.nan
                    
            try:
                df_ecg.loc[pos, 'Site'] = ecg_list[element].TestDemographics["Site"]
            except:
                df_ecg.loc[pos, 'Site'] = np.nan
            
            try:
                df_ecg.loc[pos, 'SiteName'] = ecg_list[element].TestDemographics["SiteName"]
            except:
                df_ecg.loc[pos, 'SiteName'] = np.nan
                
            try:
                df_ecg.loc[pos, 'RoomID'] = ecg_list[element].TestDemographics["RoomID"]
            except:
                df_ecg.loc[pos, 'RoomID'] = np.nan
                
            try:
                df_ecg.loc[pos, 'AcquisitionDevice'] = ecg_list[element].TestDemographics["AcquisitionDevice"]
            except:
                df_ecg.loc[pos, 'AcquisitionDevice'] = np.nan
                
            try:
                df_ecg.loc[pos, 'Status'] = ecg_list[element].TestDemographics["Status"]
            except:
                df_ecg.loc[pos, 'Status'] = np.nan
            
            try:
                df_ecg.loc[pos, 'Location'] = ecg_list[element].TestDemographics["Location"]
            except:
                df_ecg.loc[pos, 'Location'] = np.nan
                
            try:
                df_ecg.loc[pos, 'LocationName'] = ecg_list[element].TestDemographics["LocationName"]    
            except:
                df_ecg.loc[pos, 'LocationName'] = np.nan
                
            try:
                df_ecg.loc[pos, 'AcquisitionTime'] = ecg_list[element].TestDemographics["AcquisitionTime"]
            except:
                df_ecg.loc[pos, 'AcquisitionTime'] = np.nan
            
            try:
                df_ecg.loc[pos, 'AcquisitionDate'] = ecg_list[element].TestDemographics["AcquisitionDate"]
            except:
                df_ecg.loc[pos, 'AcquisitionDate'] = np.nan
        
        # Get date and time in correct format
        df_ecg.loc[:,"AcquisitionDateTime"] = pd.to_datetime(df_ecg.AcquisitionDate +
                                                    " " +
                                                    df_ecg.AcquisitionTime,
                                                    dayfirst=True)
        
        df_ecg.drop(["AcquisitionDate", "AcquisitionTime"], inplace = True, axis = 1)
        
        # Extract waveform data into multidimensional numpy array
        #Some samples have more than 2500 time points, for example 5000, we
        # only keep the first 2500 such that all sampels have identical lengths
        df_ecg_waveform = np.array([np.transpose(np.array(list(
            i.LeadVoltagesRhythm.values()))[:,:2500]) for i in ecg_list]) 
        
        # Pre-process ECG waveform data
        if preprocess:
            x_ecg = np.array(
                [preprocess_ecg(i, 500, leads = "all_leads",
                                remove_baseline = True) for i in df_ecg_waveform])
    
        # Min-Max scale the ECG waveforms
        # scaler = MinMaxScaler()
    
        # Standardisation scaling of the ECG Waverforms
        if scale:
            scaler = StandardScaler()
            
            x_ecg_shape = x_ecg.shape # Determine shape
            
            x_ecg = np.reshape(
                scaler.fit_transform(np.reshape(
                    x_ecg,(x_ecg_shape[0] * 2500, 12))
                    ), x_ecg_shape)
        
        if save:
            df_ecg.to_csv(f"{data_path}/ECG_metadata.csv")
        
        return df_ecg, x_ecg, ecg_list
    

def addDamicAdmissionData(data, mostRecentDamicFile):
    # Load most recent version of DAM-IC and create extra columns to fill in the ECG file
    damic = pd.read_csv(mostRecentDamicFile)
    damic.hrStart = pd.to_datetime(damic.hrStart)
    damic.hrStop = pd.to_datetime(damic.hrStop)
    data["AcquisitionDateTime"] = pd.to_datetime(data["AcquisitionDateTime"])

    data["isBeforeIcu"]=0
    data["isAfterIcu"]=0
    data["isDuringIcu"]=0
    data["timeToNextIcu"]=pd.NaT
    data["timeSincePrevIcu"]=pd.NaT

    # Add empty column for unique encounter ID, isDuring24Hr
    data["uniqueEncId"] = pd.NA
    data["isDuring24HrICU"] = 0

    #%% Loop over all DAM-IC admissions to check if an ECG was measured before,
    # during or after an ICU stay
    damicHeight=len(damic)
    for ind in range(damicHeight):    
        thisLft = damic.loc[ind,"lifetimeNumber"]
        inDateTime = damic.loc[ind,"hrStart"]
        outDateTime = damic.loc[ind,"hrStop"]
        
        # Check if an ECG was made before, during or after ICU stay
        thisPat = data.PatientID == thisLft            
        isAfter = thisPat & (data.AcquisitionDateTime > outDateTime)
        isBefore = thisPat & (data.AcquisitionDateTime < inDateTime)
        isDuring = thisPat & (data.AcquisitionDateTime >= inDateTime) & \
            (data.AcquisitionDateTime <= outDateTime)
        isDuring24Hr = thisPat & (data.AcquisitionDateTime >= inDateTime) & \
                       (data.AcquisitionDateTime <= inDateTime + pd.Timedelta(hours=24))
            
        data.loc[isBefore,"isBeforeIcu"]=1
        data.loc[isAfter,"isAfterIcu"]=1
        data.loc[isDuring,"isDuringIcu"]=1

        # Set uniqueEncId for rows where isDuring is True
        data.loc[isDuring, "uniqueEncId"] = damic.loc[ind, "uniqueEncId"]
        data.loc[isDuring24Hr, "isDuring24HrICU"] = 1

        # Print update status in console as this can take a while:
        if (np.mod(ind,100) == 0) | (ind==damicHeight):
            print('\033[1A', end='\x1b[2K') # Replaces previous line instead of addding a new one
            print("Admission "+str(ind)+"/"+str(damicHeight), end='\r')

    #%% Loop over all ECGs to add the time until the next ICU admission, or the time 
    # past since the previous admission. If this ECG was measured during an admission
    # these times are set to 0
    dataHeight=len(data)
    zeroTimeDelta = pd.Timedelta(0)
    for ind in range(dataHeight):    
        if sum((data.loc[ind,"isBeforeIcu"],data.loc[ind,"isAfterIcu"],data.loc[ind,"isDuringIcu"]))==0:
            continue # because ECG is not before, during or after, most likely because patient is not in ADAMIC
        elif data.loc[ind,"isDuringIcu"]==1: # No need to calculate if this is during an ICU stay
            data.loc[ind,"timeToNextIcu"] = zeroTimeDelta
            data.loc[ind,"timeSincePrevIcu"] = zeroTimeDelta
            continue
            
        ecgPatId = data.loc[ind,"PatientID"]    
        ecgDateTime = data.loc[ind,"AcquisitionDateTime"]
        
        tmpTable = damic[damic.lifetimeNumber==ecgPatId]
        
        # Calculate the time differences for time to and time since ICU admission
        diff = tmpTable.hrStop-ecgDateTime # Use hrStop for time since
        timeDifs = [n for n in diff if n<zeroTimeDelta]
        timeDifs.append(pd.NaT) # Append NaT for when no values are present
        data.loc[ind,"timeSincePrevIcu"] = max(timeDifs)
        
        diff = tmpTable.hrStop-ecgDateTime # Use hrStart for time to next
        timeDifs = [n for n in diff if n>zeroTimeDelta]
        timeDifs.append(pd.NaT) # Append NaT for when no values are present
        data.loc[ind,"timeToNextIcu"] = max(timeDifs)

        
        # Print update status in console as this can take a while:
        if (np.mod(ind,100) == 0) | (ind==dataHeight):
            print('\033[1A', end='\x1b[2K') # Replaces previous line instead of addding a new one
            print("ECG "+str(ind)+"/"+str(dataHeight), end='\r')#
            
    # Function to convert timedelta string to hours
    def convert_to_hours(duration):
        if isinstance(duration, pd.Timedelta):
            # Direct conversion for Timedelta objects
            total_hours = duration.total_seconds() / 3600
            return total_hours
        
        if pd.isna(duration) or duration == 'NaT':
            return np.nan  # Handle NaT as NaN for numerical calculations
        
        # Match the pattern and extract the days, hours, minutes, and seconds
        pattern = r'(?P<sign>-)?(?P<days>\d+) days (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)'
        match = re.match(pattern, duration)
        
        if match:
            sign = -1 if match.group('sign') else 1
            days = int(match.group('days'))
            hours = int(match.group('hours'))
            minutes = int(match.group('minutes'))
            seconds = int(match.group('seconds'))
            
            # Calculate total hours
            total_hours = sign * (days * 24 + hours + minutes / 60 + seconds / 3600)
            return total_hours
        else:
            return np.nan
    
    # Apply the function to the column
    data['timeToNextIcu_hours'] = data['timeToNextIcu'].apply(convert_to_hours)
    data['timeSincePrevIcu_hours'] = data['timeSincePrevIcu'].apply(convert_to_hours)
            
    return data   
