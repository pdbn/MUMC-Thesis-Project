# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:39:01 2023

@author: G10062682
"""

import os
import csv
import array
import base64
import xmltodict

import numpy as np



class ECGXMLReader:
    """ Extract voltage data from a ECG XML file """
    def __init__(self, path, augmentLeads=False):
        try: 
            with open(path, 'rb') as xml:
                self.ECG = xmltodict.parse(xml.read().decode('utf8'))
            
            self.augmentLeads           = augmentLeads
            self.path                   = path

            self.PatientDemographics    = self.ECG['RestingECG']['PatientDemographics']
            self.TestDemographics       = self.ECG['RestingECG']['TestDemographics']
            self.RestingECGMeasurements = self.ECG['RestingECG']['RestingECGMeasurements']
            self.Waveforms              = self.ECG['RestingECG']['Waveform']
            
            self.LeadVoltagesRhythm     = self.makeLeadVoltagesRhythm()
            self.LeadVoltagesMedian     = self.makeLeadVoltagesMedian()            
            
            
        
        except Exception as e:
            print(str(e))
    
    def makeLeadVoltagesRhythm(self):

        num_leads = 0

        leads = {}

        for lead in self.Waveforms[1]['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_b64  = base64.b64decode(lead_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[lead['LeadID']] = lead_vals
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
            
        return leads
            
    def makeLeadVoltagesMedian(self):

        num_leads = 0

        leads = {}

        for lead in self.Waveforms[0]['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_b64  = base64.b64decode(lead_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[lead['LeadID']] = lead_vals
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
        
        return leads

    def getLeadVoltagesMedian(self, LeadID):
        return self.LeadVoltagesMedian[LeadID]
    
    def getLeadVoltagesRhythm(self, LeadID):
        return self.LeadVoltagesRhythm[LeadID]
    
    def getAllVoltages(self):
        return self.LeadVoltagesMedian, self.LeadVoltagesRhythm