# Candidate Number: 1103298
# Github: https://github.com/S2412153/GRA4152/blob/main/testing_gml.py

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize
from abc import ABC, abstractmethod
import statsmodels.api as sm
import argparse

# Superclass for DataLoader and its subclasses
class DataLoader(ABC):
    def __init__(self):
        self._X = None  # Placeholder for predictor variables
        self._y = None  # Placeholder for response variable

    # Abstract method to load data for different GMLs
    @abstractmethod
    def load_data(self):
        pass

    # Method to add a constant column to predictors
    def add_constant(self):
        if self._X is None:
            raise ValueError("Data not loaded. Load data before adding a constant.")
        self._X = sm.add_constant(self._X)
        print("Added constant column to X.")

    # Getter for X
    @property
    def x(self):
        if self._X is None:
            raise ValueError("Data not loaded. Load data before accessing X.")
        return self._X

    # Getter for y
    @property
    def y(self):
        if self._y is None:
            raise ValueError("Data not loaded. Load data before accessing y.")
        return self._y
    
    # Transpose property for X
    @property
    def x_transpose(self):
        if self._X is None:
            raise ValueError("Data not loaded. Load data before accessing X.")
        
        # Ensure the transpose matches statsmodels' expectations: [N, p+1]
        X_t = self._X.T
        
        # Log a warning instead of asserting
        if X_t.shape[0] <= X_t.shape[1]:
            print(f"Warning: Transposed shape {X_t.shape} may not match expectations [N, p+1].")
        return X_t

    # Method to check if X and y have matching dimensions
    def check_shape(self):
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Please load data before checking shapes.")
        if self._X.shape[0] != len(self._y):
            raise ValueError(
                f"Shape mismatch: X has {self._X.shape[0]} rows but y has {len(self._y)} elements."
            )

# DataLoader subclass for Statsmodels datasets
class StatsModelsLoader(DataLoader):
    def __init__(self, dataset_name, response_var, predictors=None):
        super().__init__()
        self.dataset_name = dataset_name  # Name of the dataset
        self.response_var = response_var  # Response variable
        self.predictors = predictors      # Predictor variables (optional)

    def load_data(self):
        # fetch dataset based on the specified name using statsmodel library
        # we store the data into pandas datasets for easier variable manipulation
        if self.dataset_name.lower() == "duncan":
            dataset = sm.datasets.get_rdataset("Duncan", "carData").data
        elif self.dataset_name.lower() == "spector":
            dataset = sm.datasets.spector.load_pandas().data
        else:
            dataset = sm.datasets.get_rdataset(self.dataset_name).data # a generic approach

        # Check if response_var exists in dataset
        if self.response_var not in dataset.columns:
            raise ValueError(f"Response variable '{self.response_var}' not found in dataset.")

        # Set predictors and response variable
        self._y = dataset[self.response_var]
        if self.predictors:
            missing_predictors = [p for p in self.predictors if p not in dataset.columns] # check if given predictor exists in dataset column
            if missing_predictors:
                raise ValueError(f"Predictors {missing_predictors} not found in dataset.") # raise error if predictor doesn't exist
            self._X = dataset[self.predictors]

        # Default to all columns except response variable
        else:
            self._X = dataset.drop(columns=[self.response_var])
            # if predictor not defined, use all columns except response variable as self._X
            if self._X.empty:
                raise ValueError("No predictors specified, and dropping the response variable resulted in an empty dataset.")
            print("Using all available predictors.")

        print(f"Loaded dataset: {self.dataset_name}")
        print(f"Predictors: {self._X.columns.tolist()}")
        print(f"Response: {self.response_var}")

        self.check_shape()
        print(f"Successfully loaded and validated dataset '{self.dataset_name}'.")

# DataLoader subclass for loading CSV files from the internet
class InternetCSVLoader(DataLoader):
    def __init__(self, url, response_var, predictors=None, dataset_name=None):
        super().__init__()
        self.url = url  # URL to fetch the dataset
        self.response_var = response_var  # Response variable
        self.predictors = predictors      # Predictor variables
        self.dataset_name = dataset_name if dataset_name else url.split('/')[-1]  # Dataset name derived from URL if not provided

    def load_data(self):
        # Read dataset from the provided URL
        dataset = pd.read_csv(self.url)

        # Ensure response_var exists
        if self.response_var not in dataset.columns:
            raise ValueError(f"Response variable '{self.response_var}' not found in the dataset from {self.url}.")

        # Set predictors and response variable
        self._y = dataset[self.response_var]
        if self.predictors:
            missing_predictors = [p for p in self.predictors if p not in dataset.columns]
            if missing_predictors:
                raise ValueError(f"Predictors {missing_predictors} not found in the dataset.")
            self._X = dataset[self.predictors]
        else:
            self._X = dataset.drop(columns=[self.response_var])
            if self._X.empty:
                raise ValueError("No predictors specified, and dropping the response variable resulted in an empty dataset.")

        print(f"Loaded data from URL: {self.url}")
        print(f"Predictors: {self._X.columns.tolist()}")
        print(f"Response: {self.response_var}")

        self.check_shape()
        print("Successfully loaded and validated the internet CSV file.")

# GLM superclass and its subclasses
class GLM(ABC):
    def __init__(self, X, y):
        self.X = X  # Predictor variables
        self.y = y  # Response variable
        self.beta = None  # Placeholder for model coefficients

    # Fit the model by minimizing the negative log-likelihood (equivalent to MLE)
    def fit(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = np.zeros(self.X.shape[1])  # Default initial guess is zero
        result = minimize(lambda beta: self.neg_log_likelihood(beta, self.X, self.y), initial_guess)
        
        self.beta = result.x  # Store the optimized coefficients
        return self.beta

    # Predict response values for new data
    def predict(self, X_new=None):
        if self.beta is None:
            raise ValueError("Model is not fitted yet.")
        if X_new is None:
            X_new = self.X  # Default to using the training data
        eta = np.dot(X_new, self.beta)
        return self.link_function(eta)

    @abstractmethod
    def link_function(self, eta):
        pass

    @abstractmethod
    def neg_log_likelihood(self, beta, X, y):
        pass

# Subclass for Normal GLM
class NormalGLM(GLM):
    def link_function(self, eta):
        return eta  # Identity link function

    def neg_log_likelihood(self, beta, X, y):
        mu = self.link_function(np.dot(X, beta))
        return -np.sum(norm.logpdf(y, mu))

# Subclass for Bernoulli GLM
class BernoulliGLM(GLM):
    def __init__(self, X, y, alpha=0.0):
        super().__init__(X, y)
        self.alpha = alpha  # Regularization term to avoid overfitting

    def link_function(self, eta):
        return 1 / (1 + np.exp(-eta))  # Logistic link function

    def neg_log_likelihood(self, beta, X, y):
        eta = np.dot(X, beta)
        mu = self.link_function(eta)
        return -np.sum(y * np.log(mu + 1e-4) + (1 - y) * np.log(1 - mu + 1e-4)) + self.alpha * np.sum(beta**2)

# Subclass for Poisson GLM
class PoissonGLM(GLM):
    def link_function(self, eta):
        return np.exp(eta)  # Exponential link function

    def neg_log_likelihood(self, beta, X, y):
        # Return boolean array if y is non-negative, check all elements in the array are True. Otherwise, the condition fails.
        # Round each element to the nearest integer. Check and verify if all elements are integers.
        # This is due to non-negative and fractional values are not valid for the Poisson distribution.
        if not np.all(y >= 0) or not np.all(np.floor(y) == y):
            raise ValueError("y must contain only non-negative integers for Poisson GLM.")
        mu = self.link_function(np.dot(X, beta))
        return -np.sum(poisson.logpmf(y, mu))

# Function to test various models
def run_test(loader, model_types, add_intercept=False, show_summary=False):
    loader.load_data()  # Load data using the provided DataLoader
    if add_intercept:
        loader.add_constant()  # Add an intercept if specified

    X, y = loader.x, loader.y  # Get predictors and response variable
    results = {}

    # Iterate over specified model types
    for model_type in model_types:
        if model_type == "normal":
            model = NormalGLM(X.values, y)  # Custom Normal GLM
            sm_model = sm.GLM(y, X, family=sm.families.Gaussian())  # Statsmodels Normal GLM
        elif model_type == "bernoulli":
            model = BernoulliGLM(X.values, y, alpha=1e-4)
            sm_model = sm.GLM(y, X, family=sm.families.Binomial())
        elif model_type == "poisson":
            model = PoissonGLM(X.values, y)
            sm_model = sm.GLM(y, X, family=sm.families.Poisson())
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Fit the models and compare results
        beta_custom = model.fit()
        sm_results = sm_model.fit()
        beta_sm = sm_results.params.values

        # Make predictions
        predictions_custom = model.predict(X.values)
        predictions_sm = sm_results.predict(X)

        # Print results
        print_output(
            dataset_name=loader.dataset_name,
            predictors=X.columns.tolist(), # pandas.Index object convert to list for better readability
            response_var=loader.response_var,
            add_intercept=add_intercept,
            model_type=model_type,
            beta_custom=beta_custom,
            beta_sm=beta_sm,
            predictions_custom=predictions_custom,
            predictions_sm=predictions_sm,
        )

        if show_summary:
            print("\nStatsmodels Summary:")
            print("---------------------")
            print(sm_results.summary())
            
    return results

# Helper function to format and display results
def print_output(dataset_name, predictors, response_var, add_intercept, model_type, beta_custom, beta_sm, predictions_custom, predictions_sm):
    print("\nDataset Information:")
    print("---------------------")
    print(f"Loaded dataset: {dataset_name}")
    print(f"Predictors: {', '.join(predictors)}")
    print(f"Response Variable: {response_var}")
    print("Status: Successfully loaded and validated dataset.")
    if add_intercept:
        print("Intercept: Added to predictors.")

    print(f"\nModel Results: {model_type.capitalize()}")
    print("------------------------")
    print("Custom Betas:")
    for name, value in zip(["Intercept"] + predictors, beta_custom):
        print(f"  {name:10}: {value: .4f}")

    print("\nStatsmodels Betas:")
    for name, value in zip(["Intercept"] + predictors, beta_sm):
        print(f"  {name:10}: {value: .4f}")

    print("\nPredictions (First 10 Observations):")
    print("------------------------------------")
    print(f"Custom Model Predictions:    {[round(pred, 4) for pred in predictions_custom[:10]]}")
    print(f"Statsmodels Predictions:     {[round(pred, 4) for pred in predictions_sm[:10].tolist()]}")

# Command-line execution using argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GLM tests and compare models.")
    parser.add_argument('--model', type=str, choices=['normal', 'bernoulli', 'poisson'], nargs='+', required=True, help="Select GLM model(s) to test.")
    parser.add_argument('--dset', type=str, required=True, help="Name of dataset.")
    parser.add_argument('--predictors', nargs='+', type=str, help="List of predictors.")
    parser.add_argument('--add_intercept', action='store_true', help="Add an intercept.")
    parser.add_argument('--summary', action='store_true', help="Show model summary.")

    args = parser.parse_args()

    if args.dset.lower() == "duncan":
        loader = StatsModelsLoader("Duncan", response_var="income", predictors=args.predictors)
    elif args.dset.lower() == "spector":
        loader = StatsModelsLoader("spector", response_var="GRADE", predictors=args.predictors)
    elif args.dset.lower() == "warpbreaks":
        loader = InternetCSVLoader("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv",
                                   response_var="breaks", predictors=args.predictors)
    else:
        raise ValueError("Unsupported dataset.")

    run_test(loader, model_types=args.model, add_intercept=args.add_intercept, show_summary=args.summary)
