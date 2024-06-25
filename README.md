# OssicoNN

OssicoNN is a conditional Invertible Neural Network (cINN) code designed to process stellar high-resolution spectroscopy data from GIRAFFE instruments using the FrEIA framework.


Utilizing the flux uncertainty present in the spectra and the properties of cINN, \texttt{OssicoNN} offers an estimation of the overall uncertainty derived during in the inference process, encompassing both the epistemic error associated with the neural network and the aleatoric error intrinsic to the data. \texttt{OssicoNN} is trained on stellar observational data from GIRAFFE dataset of the Gaia-ESO Survey, to  derive effective temperature, surface gravity, metallicity, and various elemental abundances (Aluminum, Magnesium, Calcium, Nickel, Titanium, and Silicon) from stellar spectra. 

This repository includes the following files:
* conditioning_ossiconn.py 
* conditional_ossiconn.py 

Hyperparameters used in the model are available on demand.

For more information on OssicoNN, please refer to the article describing OssicoNN : toBeAdd 

For more information on Freia, visit the Freia website: https://github.com/vislearn/FrEIA 
