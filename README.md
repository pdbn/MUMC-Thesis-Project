# Master Thesis Project @ Maastricht UMC+ Hospital

## Details
This thesis internship is finished under the supervision of Jip de Kok, Frank Rosmalen, and Prof. Stephan Smeekes as part of Msc E&OR curriculum. 
🔗 [Public Version of Thesis](https://drive.google.com/file/d/1KujrOS4plZcCdR7IS9wSLKqaREbOy6aY/view?usp=sharing)

## Repository Structure
Here is the repository structure: 
```text
ECG loading scripts > scripts/
├── ECGXMLReader/
├── ECG_preprocessing/
├── data_functions/
├── visualisation/
├── vrae
```
ECGXMLReader: Contains all functions required to read ECG signals from XML files
ECG_preprocessing: Contains all functions required for preprocessing, including: augmenting leads, filters, sampling
data_functions: Contains all functions to read and process relevant files, create relevant dataframes containing metadata and ECG signals 
visualisation: Contains all functions for visualization purposes
vrae: Contains all function for the main VRAE framework
