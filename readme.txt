requirements

setup_instructions
Should I split the test set and include it?

Organization
1. Initial data analysis in dataAnalysis.ipynb

2. SVM hyperparameter tuning in SVM.ipynb

3. MLP model architectures defined in MLPfunctions.py. Additional hyperparameters for the smaller models are explored in
MLP.ipynb. Specifically this is where different learning rates and optimizers are tested.

4. The gpu training folder contains .py files for training the SVM and MLPs on a GPU. Specifically they were trained using 
City University's HPC (NVIDIA A100). This was necessary as the HIGGs boson dataset contatins a large amount of data, and increasing
the number of samples used improved the performance of the models.
After training the output files are saved as ***** and *****. The trained models are saved
in the model's folder. The training and testing losses for the MLP models are saved in the training_outputs folder as pickle files.

5. The figures in the paper and testing results can be easily reproduced using the figures_and_tests.ipynb. 