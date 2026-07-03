import numpy as np

def load_data(n_samples=100, noise_std=5.0, random_state=42): #100 samples, noise set to 5
    np.random.seed(random_state)    #ensures that the generated noise stays the same each time for the same seed

    X = np.linspace(0, 10, n_samples).reshape(-1, 1)    #creates evenly spaced points between 0 and 10, reshape converts it into a 2D coloumn matrix of the form (n_samples, n_features)
                                                        #thus X.shape becomes (100, 1) matrix

    noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples) # generates Gaussian noise with mean=0 and SD=5

    true_slope = 4.0 #defining the true slope and true intercept
    true_intercept = 3.0

    y = true_intercept + true_slope*X.flatten() + noise #builds the target values

    return X, y





from sklearn.datasets import load_diabetes
import numpy as np

def load_multifeature_data():
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


