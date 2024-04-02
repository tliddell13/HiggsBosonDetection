import pandas as pd

dataset = pd.read_csv('HIGGSdata.csv')

labels = ['class', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi', 
          'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 
          'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 
          'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
dataset.columns = labels

# Set aside 80% of the data for training and 20% for testing
train = dataset.sample(frac=0.8, random_state=200)
test = dataset.drop(train.index)
# Save the training and testing data to separate files
train.to_csv('HIGGS_train.csv', index=False)
test.to_csv('HIGGS_test.csv', index=False)
