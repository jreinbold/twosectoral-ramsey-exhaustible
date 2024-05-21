# twosectoral-ramsey-exhaustible

This project contains code to numerically simulate a two-sector Ramsey model with asymmetric sectors, considering an exhaustible resource input in one sector. I wrote the code for my Master's thesis to evaluate the distributional consequences of the climate transformation. The main file, model.py, is a script to generate the model solution and specific output tables and plots. The repository is organized as follows:

As mentioned, "model.py" is a script to solve the two-sector Ramsey model numerically derived in my Master's thesis. It defines a class to initialize the model under certain parameter choices. Furthermore, it defines a function "simulate" that can be performed for the class. The function returns the model solution, including time series data of the variables as well as specific values to compute a welfare analysis between different parameterizations. Additionally, it contains code to perform the welfare analysis by determining the consumption-equivalent variation. Finally, it illustrates and stores the results.

The folder "distributions" contains random generators for different distributional setups that I referred to as "synthetic economies" in my Master's thesis. Morevoer, it contains the specific datasets that I used for the simulation in my Master's thesis as CSV files.

The file "steadystate_calculation.py" contains auxiliary code to iteratively identify valid initial guesses for solving of the models in "model.py".

Do not hesitate to contact me if you have any questions regarding the code or the two-sector Ramsey model.
