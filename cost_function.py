import numpy as np

def mean_squared_error(y_true,y_predicted):
    # Mean Squared error function
    
    # Given values 
    y_true # y_true = y
    
    # Calculated values 
    y_predicted # y_pred = y^
    
    # Mean Squared Error
    # Mean = sum of the terms / number of terms
    
    mse = np.square(np.subtract(y_true,y_predicted)).mean()
    return mse


#out = mean_squared_error([1],[0.6])
#print(out)
