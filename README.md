# Master Thesis Project @ Maastricht UMC+ Hospital

## Details
This thesis internship is finished under the supervision of Jip de Kok, Frank Rosmalen, and Prof. Stephan Smeekes as part of Msc E&OR curriculum.  <br>
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
- ECGXMLReader: Contains all functions required to read ECG signals from XML files <br>
- ECG_preprocessing: Contains all functions required for preprocessing, including: augmenting leads, filters, sampling <br>
- data_functions: Contains all functions to read and process relevant files, create relevant dataframes containing metadata and ECG signals  <br>
- visualisation: Contains all functions for visualization purposes <br>
- vrae: Contains all functions for the main VRAE framework

## My VRAE framework: 
<img width="1800" height="562" alt="vrae" src="https://github.com/user-attachments/assets/07aec9bb-3136-40cf-9393-648367a2ecb3" />
_illustrated by me_

The VRAE architecture consists of 3 principal components: the encoder, the latent space, and the decoder.<br>
Inpput: Sequential data<br>
Output: Reconstructed sequential data

- Encoder processes an input sequence through an RNN, where each cell receives the current input x_t and the previous hidden state h_{t-1}. The RNN cell, which can be either LSTM or GRU depending on the design choice, produces a sequence of hidden states, from which the final hidden state is extracted as a compact representation of the entire input sequence.
- Latent space: The h_end is projected through two parallel fully connected layers to obtain the parameters of the approximate posterior distribution:  mean, standard deviation. T
- Note: To allow gradient-based optimization, the reparameterization trick is applied.
- Decoder is an RNN that reconstructs the original sequence from the latent representation.
