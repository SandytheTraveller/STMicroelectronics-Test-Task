# Attempt to unpickle the regressor
import pickle

with open('/Users/aleksandrak/Desktop/STMicroelectronics-Test-Task/output_inverse.ini/LRRidge.pickle', 'rb') as f:
    loaded_regressor = pickle.load(f)
    print(loaded_regressor._regressor)