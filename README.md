nWTA.git
__________________________________________________________
### Code to reproduce the main findings presented in the paper
### "Robust parallel decision-making in neural circuits with nonlinear inhibition"
### by B. Kriener, R. Chaudhuri and Ila R. Fiete
__________________________________________________________


#### nwta_dyn_fcs.py:

Function definitions for basic winner-take-all dynamics with nonlinear inhibition. 
Is needed for the example script nwta_dyn_examples.py

#### nwta_dyn_examples.py:

Plots activation and rate traces.
Example of self-terminating nWTA (prints decision time and if result is accurate).

#### WTA_stability.py

Computes breakdown of WTA as a function of noise amplitude sigma. 
In particular: 'kink' in solid black curve corresponds to noise amplitude at which the non-WTA branches come into existence, see Supplementary Appendix S2.6.
