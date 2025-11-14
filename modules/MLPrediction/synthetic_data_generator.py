"""
Synthetic Data Generation for EV Charging Demand Prediction

This module implements advanced synthetic data generation techniques including:
1. Variational Autoencoder (VAE) for high-quality synthetic samples
2. Gaussian Mixture Model (GMM) based augmentation
3. Targeted minority oversampling for imbalanced geospatial data

Author: AI Assistant
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ctgan import CTGAN
try:
    from imblearn.over_sampling import SMOTE, SVMSMOTE
except ImportError:
    try:
        from imbalanced_learn.over_sampling import SMOTE, SVMSMOTE
    except ImportError:
        print("Warning: imblearn not available, SMOTE will be disabled")
        SMOTE = None
        SVMSMOTE = None

logger = logging.getLogger(__name__)

class VariationalAutoencoder:
    """
    Variational Autoencoder for generating synthetic geospatial demand data.
    
    This VAE is specifically designed for tabular geospatial data with:
    - Spatial coordinates (lat, lon)
    - Urban features (POI densities, accessibility metrics)
    - Target demand scores
    """
    
    def __init__(self, latent_dim: int = 8, hidden_dims: list = [64, 32], 
                 learning_rate: float = 0.001, random_state: int = 42):
        """
        Initialize the VAE.
        
        Args:
            latent_dim: Dimensionality of the latent space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for training
            random_state: Random seed for reproducibility
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _build_encoder(self, input_dim: int):
        """Build the encoder network."""
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        return keras.Model(inputs, [z_mean, z_log_var], name='encoder')
    
    def _build_decoder(self, input_dim: int):
        """Build the decoder network."""
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        
        # Hidden layers (reverse order)
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(input_dim, activation='linear')(x)
        
        return keras.Model(latent_inputs, outputs, name='decoder')
    
    def _sampling(self, args):
        """Reparameterization trick for sampling from latent space."""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_vae(self, input_dim: int):
        """Build the complete VAE."""
        self.encoder = self._build_encoder(input_dim)
        self.decoder = self._build_decoder(input_dim)
        
        # VAE model
        inputs = keras.Input(shape=(input_dim,))
        z_mean, z_log_var = self.encoder(inputs)
        z = layers.Lambda(self._sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        outputs = self.decoder(z)
        
        self.vae = keras.Model(inputs, outputs, name='vae')
        
        # VAE loss
        reconstruction_loss = keras.losses.mse(inputs, outputs)
        reconstruction_loss *= input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, 
            validation_split: float = 0.2, verbose: int = 0) -> dict:
        """
        Train the VAE.
        
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Training VAE on {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build the VAE
        self._build_vae(X.shape[1])
        
        # Train the VAE
        history = self.vae.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_fitted = True
        logger.info("VAE training completed")
        
        return history.history
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples in original scale
        """
        if not self.is_fitted:
            raise ValueError("VAE must be fitted before generating samples")
        
        # Sample from latent space
        z_samples = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate samples
        synthetic_scaled = self.decoder.predict(z_samples, verbose=0)
        
        # Scale back to original range
        synthetic_samples = self.scaler.inverse_transform(synthetic_scaled)
        
        logger.info(f"Generated {n_samples} synthetic samples")
        return synthetic_samples


class SyntheticDataGenerator:
    """
    Main class for synthetic data generation with multiple techniques.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.vae = None
        self.gmm = None
        self.ctgan = None
        
    def generate_vae_samples(self, X: pd.DataFrame, y: pd.Series, 
                           n_synthetic: int = None, 
                           target_percentile: float = 0.75) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using VAE, focusing on high-demand areas.
        
        Args:
            X: Feature matrix
            y: Target variable (demand scores)
            n_synthetic: Number of synthetic samples (default: 50% of high-demand samples)
            target_percentile: Percentile threshold for high-demand areas
            
        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        logger.info("Generating synthetic samples using VAE...")
        
        # Identify high-demand samples
        threshold = y.quantile(target_percentile)
        high_demand_mask = y >= threshold
        
        X_high = X[high_demand_mask]
        y_high = y[high_demand_mask]
        
        if len(X_high) < 10:
            logger.warning("Too few high-demand samples for VAE training. Using all data.")
            X_high = X
            y_high = y
        
        # Determine number of synthetic samples
        if n_synthetic is None:
            n_synthetic = max(50, len(X_high) // 2)
        
        # Combine features and target for VAE training
        combined_data = np.column_stack([X_high.values, y_high.values])
        
        # Initialize and train VAE
        self.vae = VariationalAutoencoder(
            latent_dim=min(8, combined_data.shape[1] // 2),
            hidden_dims=[64, 32],
            random_state=self.random_state
        )
        
        self.vae.fit(combined_data, epochs=100, verbose=0)
        
        # Generate synthetic samples
        synthetic_combined = self.vae.generate(n_synthetic)
        
        # Split back into features and target
        synthetic_X = pd.DataFrame(
            synthetic_combined[:, :-1], 
            columns=X.columns
        )
        synthetic_y = pd.Series(synthetic_combined[:, -1], name=y.name)
        
        # Ensure realistic bounds
        synthetic_X = synthetic_X.clip(
            lower=X.quantile(0.01), 
            upper=X.quantile(0.99)
        )
        synthetic_y = synthetic_y.clip(
            lower=y.quantile(0.01), 
            upper=y.quantile(0.99)
        )
        
        logger.info(f"Generated {len(synthetic_X)} synthetic samples using VAE")
        return synthetic_X, synthetic_y
    
    def generate_gmm_samples(self, X: pd.DataFrame, y: pd.Series,
                           n_synthetic: int = None,
                           target_percentile: float = 0.75) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using Gaussian Mixture Model.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_synthetic: Number of synthetic samples
            target_percentile: Percentile threshold for high-demand areas
            
        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        logger.info("Generating synthetic samples using GMM...")
        
        # Focus on high-demand samples
        threshold = y.quantile(target_percentile)
        high_demand_mask = y >= threshold
        
        X_high = X[high_demand_mask]
        y_high = y[high_demand_mask]
        
        if len(X_high) < 10:
            logger.warning("Too few high-demand samples for GMM. Using all data.")
            X_high = X
            y_high = y
        
        if n_synthetic is None:
            n_synthetic = max(25, len(X_high) // 3)
        
        # Combine features and target
        combined_data = np.column_stack([X_high.values, y_high.values])
        
        # Fit GMM
        n_components = min(5, len(X_high) // 10)  # Adaptive number of components
        self.gmm = GaussianMixture(
            n_components=max(1, n_components), 
            random_state=self.random_state
        )
        self.gmm.fit(combined_data)
        
        # Generate synthetic samples
        synthetic_combined, _ = self.gmm.sample(n_synthetic)
        
        # Split back into features and target
        synthetic_X = pd.DataFrame(
            synthetic_combined[:, :-1], 
            columns=X.columns
        )
        synthetic_y = pd.Series(synthetic_combined[:, -1], name=y.name)
        
        # Ensure realistic bounds
        synthetic_X = synthetic_X.clip(
            lower=X.quantile(0.01), 
            upper=X.quantile(0.99)
        )
        synthetic_y = synthetic_y.clip(
            lower=y.quantile(0.01), 
            upper=y.quantile(0.99)
        )
        
        logger.info(f"Generated {len(synthetic_X)} synthetic samples using GMM")
        return synthetic_X, synthetic_y
    
    def generate_noise_augmentation_samples(self, X: pd.DataFrame, y: pd.Series,
                                          n_synthetic: int = None,
                                          noise_factor: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using simple noise addition.
        
        This is a simple but reliable method that adds controlled noise to existing samples
        to create new synthetic data points.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_synthetic: Number of synthetic samples to generate
            noise_factor: Amount of noise to add (as fraction of feature std)
            
        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        logger.info("Generating synthetic samples using noise augmentation...")
        
        if n_synthetic is None:
            n_synthetic = max(50, len(X) // 2)  # Default to adding 50% more samples
        
        try:
            # Randomly select samples to augment
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(len(X), size=n_synthetic, replace=True)
            
            # Create base synthetic samples
            synthetic_X = X.iloc[sample_indices].copy().reset_index(drop=True)
            synthetic_y = y.iloc[sample_indices].copy().reset_index(drop=True)
            
            # Add controlled noise to numerical features
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    col_std = X[col].std()
                    noise = np.random.normal(0, col_std * noise_factor, len(synthetic_X))
                    synthetic_X[col] = synthetic_X[col] + noise
                    
                    # Ensure values stay within reasonable bounds
                    lower_bound = X[col].quantile(0.01)
                    upper_bound = X[col].quantile(0.99)
                    synthetic_X[col] = synthetic_X[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Add small noise to target variable
            y_std = y.std()
            y_noise = np.random.normal(0, y_std * noise_factor * 0.5, len(synthetic_y))
            synthetic_y = synthetic_y + y_noise
            
            # Ensure target values stay within bounds
            synthetic_y = synthetic_y.clip(
                lower=y.quantile(0.01),
                upper=y.quantile(0.99)
            )
            
            logger.info(f"Generated {len(synthetic_X)} synthetic samples using noise augmentation")
            return synthetic_X, synthetic_y
            
        except Exception as e:
            logger.error(f"Noise augmentation failed: {str(e)}")
            raise
    
    def generate_smote_regression_samples(self, X: pd.DataFrame, y: pd.Series,
                                        n_synthetic: int = None,
                                        target_percentile: float = 0.75) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using SMOTE for imbalanced regression.
        
        This method addresses class imbalance in regression by:
        1. Discretizing the continuous target into bins
        2. Applying SMOTE to oversample minority classes
        3. Reconstructing continuous target values
        
        Args:
            X: Feature matrix
            y: Target variable (continuous)
            n_synthetic: Number of synthetic samples to generate
            target_percentile: Percentile to define high-demand (minority) class
            
        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        logger.info("Generating synthetic samples using SMOTE for imbalanced regression...")
        
        if n_synthetic is None:
            n_synthetic = max(50, len(X) // 3)  # Default to adding 33% more samples
        
        try:
            # Step 1: Discretize the continuous target into bins for SMOTE
            # Use quantile-based binning to create balanced representation
            n_bins = min(5, max(2, len(y) // 10))  # Ensure at least 2 bins
            
            # Create bins with equal frequency (quantile-based)
            try:
                bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            except ValueError as e:
                # Fallback to regular binning if qcut fails
                logger.warning(f"qcut failed: {e}. Using regular binning.")
                bins = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
            
            # Remove any NaN bins
            valid_mask = ~pd.isna(bins)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            bins_valid = bins[valid_mask]
            
            if len(np.unique(bins_valid)) < 2:
                logger.warning("Target variable has insufficient variation for SMOTE. Using original data.")
                return X.copy(), y.copy()
            
            # Step 2: Apply SMOTE to oversample minority classes
            # Calculate desired samples per class
            bin_counts = pd.Series(bins_valid).value_counts()
            max_samples = min(bin_counts.max() * 2, len(X_valid) + n_synthetic)  # Cap maximum
            
            # Create balanced sampling strategy
            sampling_strategy = {}
            for bin_label in bin_counts.index:
                current_count = bin_counts[bin_label]
                desired_count = min(max_samples, current_count + n_synthetic // len(bin_counts))
                if desired_count > current_count:
                    sampling_strategy[bin_label] = desired_count
            
            if not sampling_strategy:
                logger.warning("No minority classes identified for SMOTE. Using original data.")
                return X.copy(), y.copy()
            
            # Apply SMOTE with careful parameter selection
            k_neighbors = min(4, len(X_valid) // 2 - 1, min(sampling_strategy.values()) - 1)
            if k_neighbors < 1:
                logger.warning("Not enough samples for SMOTE. Using original data.")
                return X.copy(), y.copy()
            
            if SMOTE is None:
                logger.warning("SMOTE not available. Using original data.")
                return X.copy(), y.copy()
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k_neighbors
            )
            
            X_resampled, bins_resampled = smote.fit_resample(X_valid, bins_valid)
            
            # Step 3: Reconstruct continuous target values
            # Create a mapping from bin labels to continuous values
            bin_stats = {}
            for bin_label in np.unique(bins_valid):
                mask = bins_valid == bin_label
                bin_values = y_valid[mask]
                if len(bin_values) > 0:
                    bin_stats[bin_label] = {
                        'mean': bin_values.mean(),
                        'std': bin_values.std() if len(bin_values) > 1 else y_valid.std() * 0.1,
                        'min': bin_values.min(),
                        'max': bin_values.max()
                    }
                else:
                    bin_stats[bin_label] = {
                        'mean': y_valid.mean(),
                        'std': y_valid.std() * 0.1,
                        'min': y_valid.min(),
                        'max': y_valid.max()
                    }
            
            # Reconstruct continuous target values
            y_resampled = np.zeros(len(bins_resampled))
            for i, bin_label in enumerate(bins_resampled):
                stats = bin_stats[bin_label]
                # Add controlled noise
                noise = np.random.normal(0, stats['std'] * 0.3)
                value = stats['mean'] + noise
                # Ensure value is within reasonable bounds
                y_resampled[i] = np.clip(value, stats['min'] * 0.8, stats['max'] * 1.2)
            
            # Step 4: Extract only the synthetic samples
            n_original = len(X_valid)
            synthetic_X = pd.DataFrame(X_resampled[n_original:], columns=X.columns)
            synthetic_y = pd.Series(y_resampled[n_original:], name=y.name)
            
            # Apply realistic bounds based on original data
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    lower_bound = X[col].quantile(0.01)
                    upper_bound = X[col].quantile(0.99)
                    synthetic_X[col] = synthetic_X[col].clip(lower=lower_bound, upper=upper_bound)
            
            synthetic_y = synthetic_y.clip(
                lower=y.quantile(0.01), 
                upper=y.quantile(0.99)
            )
            
            logger.info(f"Generated {len(synthetic_X)} synthetic samples using SMOTE for regression")
            return synthetic_X, synthetic_y
            
        except Exception as e:
            logger.error(f"SMOTE regression failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def generate_ctgan_samples(self, X: pd.DataFrame, y: pd.Series,
                             n_synthetic: int = None,
                             epochs: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using CTGAN (Conditional Tabular GAN).
        
        CTGAN is specifically designed for tabular data and excels at:
        - Handling mixed data types (categorical and continuous)
        - Preserving complex relationships between features
        - Generating high-quality synthetic tabular data
        
        Args:
            X: Feature matrix
            y: Target variable
            n_synthetic: Number of synthetic samples to generate
            epochs: Number of training epochs for CTGAN
            
        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        logger.info("Generating synthetic samples using CTGAN...")
        
        if n_synthetic is None:
            n_synthetic = len(X)  # Generate as many samples as original data
        
        # Combine features and target for CTGAN training
        combined_data = pd.concat([X, y], axis=1)
        
        # Clean data for CTGAN
        logger.info(f"Preparing data: {combined_data.shape}")
        logger.info(f"Data types: {combined_data.dtypes.value_counts()}")
        
        # Handle any infinite or very large values
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        combined_data = combined_data.fillna(combined_data.mean())
        
        # Ensure minimum data size for CTGAN
        if len(combined_data) < 10:
            raise ValueError(f"CTGAN requires at least 10 samples, got {len(combined_data)}")
        
        # Reset index to avoid CTGAN issues with non-sequential indices
        combined_data = combined_data.reset_index(drop=True)
        
        # Identify categorical columns more carefully
        categorical_features = []
        for col in combined_data.columns:
            if combined_data[col].dtype == 'object':
                categorical_features.append(col)
            elif combined_data[col].dtype in ['int64', 'int32'] and combined_data[col].nunique() <= 10:
                # Convert small integer columns to categorical
                combined_data[col] = combined_data[col].astype('category').astype(str)
                categorical_features.append(col)
        
        logger.info(f"Identified categorical features: {categorical_features}")
        
        try:
            # Initialize CTGAN with carefully calculated parameters
            # Calculate PAC size first (must be small enough)
            pac_size = min(5, max(1, len(combined_data) // 10))
            
            # Calculate batch size (must be even and divisible by PAC)
            base_batch_size = max(10, min(64, len(combined_data) // 2))
            
            # Ensure batch size is even and divisible by PAC
            batch_size = base_batch_size
            if batch_size % 2 != 0:
                batch_size += 1
            
            # Ensure batch size is divisible by PAC
            if batch_size % pac_size != 0:
                batch_size = ((batch_size // pac_size) + 1) * pac_size
                if batch_size % 2 != 0:
                    batch_size += pac_size
            
            # Final safety check - ensure we don't exceed data size
            batch_size = min(batch_size, len(combined_data))
            if batch_size % 2 != 0:
                batch_size -= 1
            if batch_size < 2:
                batch_size = 2
                
            logger.info(f"CTGAN parameters: batch_size={batch_size}, pac={pac_size}, data_size={len(combined_data)}")
            
            self.ctgan = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                generator_lr=1e-4,     # Conservative learning rate
                discriminator_lr=1e-4,
                generator_decay=1e-6,  # L2 regularization
                discriminator_decay=1e-6,
                pac=pac_size,
                verbose=False  # Disable for cleaner logs
            )
            
            # Train CTGAN with proper categorical columns
            logger.info(f"Training CTGAN on {len(combined_data)} samples with batch_size={batch_size}...")
            self.ctgan.fit(combined_data, discrete_columns=categorical_features)
            
            # Generate synthetic samples
            logger.info(f"Generating {n_synthetic} synthetic samples...")
            synthetic_combined = self.ctgan.sample(n_synthetic)
            
            # Ensure we have the right columns
            if not all(col in synthetic_combined.columns for col in combined_data.columns):
                missing_cols = set(combined_data.columns) - set(synthetic_combined.columns)
                raise ValueError(f"CTGAN output missing columns: {missing_cols}")
            
            # Split back into features and target
            synthetic_X = synthetic_combined[X.columns].copy()
            synthetic_y = synthetic_combined[y.name].copy()
            
            # Apply realistic bounds based on original data quantiles (only for numeric columns)
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64'] and synthetic_X[col].dtype in ['float64', 'int64']:
                    lower_bound = X[col].quantile(0.01)  # More conservative bounds
                    upper_bound = X[col].quantile(0.99)
                    synthetic_X[col] = synthetic_X[col].clip(lower=lower_bound, upper=upper_bound)
                elif col in categorical_features:
                    # For categorical columns, ensure values are in the original set
                    valid_values = X[col].unique()
                    mask = synthetic_X[col].isin(valid_values)
                    if not mask.all():
                        # Replace invalid values with random valid values
                        invalid_indices = ~mask
                        replacement_values = np.random.choice(valid_values, size=invalid_indices.sum())
                        synthetic_X.loc[invalid_indices, col] = replacement_values
            
            # Ensure target values are realistic
            y_lower = y.quantile(0.01)
            y_upper = y.quantile(0.99)
            synthetic_y = synthetic_y.clip(lower=y_lower, upper=y_upper)
            
            logger.info(f"Successfully generated {len(synthetic_X)} synthetic samples using CTGAN")
            return synthetic_X, synthetic_y
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"CTGAN training/generation failed: {str(e)}")
            logger.error(f"Full traceback: {error_details}")
            logger.error(f"Data shape: {combined_data.shape}")
            logger.error(f"Data dtypes: {combined_data.dtypes}")
            logger.error(f"Data sample:\n{combined_data.head()}")
            raise RuntimeError(f"CTGAN failed: {str(e)}") from e
    
    def augment_dataset(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'ctgan',
                       augmentation_ratio: float = 1.0,
                       use_smote_for_balance: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Augment the dataset with synthetic samples.
        
        Args:
            X: Original feature matrix
            y: Original target variable
            method: Primary augmentation method ('ctgan', 'vae', 'gmm', 'smote', 'mixed', or 'ctgan_smote')
            augmentation_ratio: Ratio of synthetic to original samples (for overall data increase)
            use_smote_for_balance: Whether to apply SMOTE for addressing imbalance (additional to main method)
            
        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        logger.info(f"Augmenting dataset using {method} method with ratio {augmentation_ratio}...")
        
        n_original = len(X)
        n_synthetic_total = int(n_original * augmentation_ratio)
        
        # Step 1: Primary data augmentation for overall increase
        if method == 'ctgan':
            synthetic_X, synthetic_y = self.generate_ctgan_samples(
                X, y, n_synthetic=n_synthetic_total
            )
        elif method == 'vae':
            synthetic_X, synthetic_y = self.generate_vae_samples(
                X, y, n_synthetic=n_synthetic_total
            )
        elif method == 'gmm':
            synthetic_X, synthetic_y = self.generate_gmm_samples(
                X, y, n_synthetic=n_synthetic_total
            )
        elif method == 'smote':
            synthetic_X, synthetic_y = self.generate_smote_regression_samples(
                X, y, n_synthetic=n_synthetic_total
            )
        elif method == 'noise':
            synthetic_X, synthetic_y = self.generate_noise_augmentation_samples(
                X, y, n_synthetic=n_synthetic_total
            )
        elif method == 'ctgan_smote':
            # Combine CTGAN for overall increase + SMOTE for balance
            n_ctgan = int(n_synthetic_total * 0.7)
            n_smote = n_synthetic_total - n_ctgan
            
            try:
                synthetic_X_ctgan, synthetic_y_ctgan = self.generate_ctgan_samples(
                    X, y, n_synthetic=n_ctgan
                )
            except Exception as e:
                logger.warning(f"CTGAN failed in ctgan_smote: {e}. Using noise augmentation instead.")
                synthetic_X_ctgan, synthetic_y_ctgan = self.generate_noise_augmentation_samples(
                    X, y, n_synthetic=n_ctgan
                )
            
            try:
                synthetic_X_smote, synthetic_y_smote = self.generate_smote_regression_samples(
                    X, y, n_synthetic=n_smote
                )
            except Exception as e:
                logger.warning(f"SMOTE failed in ctgan_smote: {e}. Using noise augmentation instead.")
                synthetic_X_smote, synthetic_y_smote = self.generate_noise_augmentation_samples(
                    X, y, n_synthetic=n_smote
                )
            
            synthetic_X = pd.concat([synthetic_X_ctgan, synthetic_X_smote], ignore_index=True)
            synthetic_y = pd.concat([synthetic_y_ctgan, synthetic_y_smote], ignore_index=True)
        elif method == 'mixed':
            # Split between CTGAN (40%), SMOTE (30%), and Noise (30%) for robustness
            n_ctgan = int(n_synthetic_total * 0.4)
            n_smote = int(n_synthetic_total * 0.3)
            n_noise = n_synthetic_total - n_ctgan - n_smote
            
            results = []
            
            # Try CTGAN
            try:
                synthetic_X_ctgan, synthetic_y_ctgan = self.generate_ctgan_samples(
                    X, y, n_synthetic=n_ctgan
                )
                results.append((synthetic_X_ctgan, synthetic_y_ctgan))
            except Exception as e:
                logger.warning(f"CTGAN failed in mixed: {e}")
            
            # Try SMOTE
            try:
                synthetic_X_smote, synthetic_y_smote = self.generate_smote_regression_samples(
                    X, y, n_synthetic=n_smote
                )
                results.append((synthetic_X_smote, synthetic_y_smote))
            except Exception as e:
                logger.warning(f"SMOTE failed in mixed: {e}")
            
            # Always use noise as fallback
            synthetic_X_noise, synthetic_y_noise = self.generate_noise_augmentation_samples(
                X, y, n_synthetic=n_noise
            )
            results.append((synthetic_X_noise, synthetic_y_noise))
            
            # Combine all successful methods
            if results:
                synthetic_X_list = [x for x, y in results]
                synthetic_y_list = [y for x, y in results]
                synthetic_X = pd.concat(synthetic_X_list, ignore_index=True)
                synthetic_y = pd.concat(synthetic_y_list, ignore_index=True)
            else:
                # Should never happen due to noise fallback, but just in case
                synthetic_X, synthetic_y = self.generate_noise_augmentation_samples(
                    X, y, n_synthetic=n_synthetic_total
                )
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'ctgan', 'vae', 'gmm', 'smote', 'noise', 'ctgan_smote', or 'mixed'")
        
        # Step 2: Optional additional SMOTE for imbalance correction (if not already done)
        if use_smote_for_balance and method not in ['smote', 'ctgan_smote', 'mixed']:
            try:
                logger.info("Applying additional SMOTE for imbalance correction...")
                extra_smote_X, extra_smote_y = self.generate_smote_regression_samples(
                    X, y, n_synthetic=max(20, n_original // 10)  # Small additional amount for balance
                )
                
                synthetic_X = pd.concat([synthetic_X, extra_smote_X], ignore_index=True)
                synthetic_y = pd.concat([synthetic_y, extra_smote_y], ignore_index=True)
                
                logger.info(f"Added {len(extra_smote_X)} SMOTE samples for balance correction")
            except Exception as e:
                logger.warning(f"Additional SMOTE failed: {e}. Continuing without balance correction.")
        
        # Combine original and synthetic data
        augmented_X = pd.concat([X, synthetic_X], ignore_index=True)
        augmented_y = pd.concat([y, synthetic_y], ignore_index=True)
        
        logger.info(f"Dataset augmented: {n_original} → {len(augmented_X)} samples ({method})")
        logger.info(f"Total synthetic samples generated: {len(synthetic_X)}")
        return augmented_X, augmented_y

