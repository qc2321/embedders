import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from .kernel import product_kernel


class ProductSpacePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, pm, max_epochs=1_000, patience=5, weights=None):
        self.pm = pm  # ProductManifold instance
        self.max_epochs = max_epochs
        self.patience = patience  # Number of consecutive epochs without improvement to consider convergence
        self.classes_ = None
        if weights is None:
            self.weights = torch.ones(len(pm.P), dtype=torch.float32)
        else:
            assert len(weights) == len(pm.P), "Number of weights must match the number of manifolds."
            self.weights = weights

    def fit(self, X, y):
        # Identify unique classes for multiclass classification
        self.classes_ = torch.unique(y).tolist()
        n_samples = X.shape[0]

        # Precompute kernel matrix
        Ks, _ = product_kernel(self.pm, X, None)
        K = torch.ones((n_samples, n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights):
            K += w * K_m

        # Store training data and labels for prediction
        self.X_train_ = X
        self.y_train_ = y

        # Initialize dictionary to store alpha coefficients for each class
        self.alpha = {}

        # For patience checking
        best_epoch, least_errors = 0, n_samples + 1

        for class_label in self.classes_:
            # One-vs-rest labels
            y_binary = torch.where(y == class_label, 1, -1)  # Shape: (n_samples,)

            # Initialize alpha coefficients for this class
            alpha = torch.zeros(n_samples, dtype=X.dtype, device=X.device)

            for epoch in range(self.max_epochs):
                # Compute decision function: f = K @ (alpha * y_binary)
                f = K @ (alpha * y_binary)  # Shape: (n_samples,)

                # Compute predictions
                predictions = torch.sign(f)

                # Find misclassified samples
                misclassified = predictions != y_binary

                # If no misclassifications, break early
                if not misclassified.any():
                    break

                # Test patience
                n_errors = misclassified.sum().item()
                if n_errors < least_errors:
                    best_epoch, least_errors = epoch, n_errors
                if epoch - best_epoch >= self.patience:
                    break

                # Update alpha coefficients for misclassified samples
                alpha[misclassified] += 1

            # Store the alpha coefficients for the current class
            self.alpha[class_label] = alpha

        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decision_values = torch.zeros((n_samples, n_classes), dtype=X.dtype, device=X.device)

        # Compute kernel matrix between training data and test data
        Ks, _ = product_kernel(self.pm, self.X_train_, X)
        K_test = torch.ones((self.X_train_.shape[0], n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights):
            K_test += w * K_m
        # K_test = self.X_train_ @ X.T

        for idx, class_label in enumerate(self.classes_):
            alpha = self.alpha[class_label]  # Shape: (n_samples_train,)
            y_binary = torch.where(self.y_train_ == class_label, 1, -1)  # Shape: (n_samples_train,)

            # Compute decision function for test samples
            f = (alpha * y_binary) @ K_test  # Shape: (n_samples_test,)
            decision_values[:, idx] = f

        return decision_values

    def predict(self, X):
        decision_values = self.predict_proba(X)
        # Return the class with the highest decision value
        argmax_idx = torch.argmax(decision_values, dim=1)
        return torch.tensor([self.classes_[i] for i in argmax_idx])
