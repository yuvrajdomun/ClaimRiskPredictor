"""
Generative AI module for data augmentation.
Creates synthetic insurance claims data to improve model training.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generates synthetic insurance claims data using generative models."""
    
    def __init__(self, latent_dim: int = 50):
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def build_generator(self, output_dim: int) -> keras.Model:
        """Build generator network."""
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_dim, activation='tanh')
        ])
        
        return model
    
    def build_discriminator(self, input_dim: int) -> keras.Model:
        """Build discriminator network."""
        
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def build_gan(self, generator: keras.Model, discriminator: keras.Model) -> keras.Model:
        """Build GAN by combining generator and discriminator."""
        
        # Make discriminator non-trainable when training generator
        discriminator.trainable = False
        
        gan_input = keras.layers.Input(shape=(self.latent_dim,))
        generated_data = generator(gan_input)
        validity = discriminator(generated_data)
        
        gan = keras.Model(gan_input, validity)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        
        return gan
    
    def prepare_training_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare and normalize data for training."""
        
        # Select numeric columns for generation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')
        
        self.feature_names = numeric_cols
        data = df[numeric_cols].values
        
        # Normalize data to [-1, 1] range for tanh activation
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_data = self.scaler.fit_transform(data)
        
        return normalized_data
    
    def train_gan(self, df: pd.DataFrame, epochs: int = 1000, batch_size: int = 32) -> Dict[str, list]:
        """Train GAN to generate synthetic insurance claims."""
        
        # Prepare data
        real_data = self.prepare_training_data(df)
        data_dim = real_data.shape[1]
        
        # Build models
        self.generator = self.build_generator(data_dim)
        self.discriminator = self.build_discriminator(data_dim)
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build and compile GAN
        self.gan = self.build_gan(self.generator, self.discriminator)
        
        # Training history
        history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
        
        # Training loop
        for epoch in range(epochs):
            # Train discriminator
            # Real data
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            real_labels = np.ones((batch_size, 1))
            
            # Fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self.generator.predict(noise, verbose=0)
            fake_labels = np.zeros((batch_size, 1))
            
            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            # Store history
            history['d_loss'].append(d_loss[0])
            history['g_loss'].append(g_loss)
            history['d_acc'].append(d_loss[1])
            
            # Print progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: D_loss: {d_loss[0]:.4f}, D_acc: {d_loss[1]:.4f}, G_loss: {g_loss:.4f}")
        
        self.is_trained = True
        logger.info("GAN training completed")
        
        return history
    
    def generate_synthetic_data(self, n_samples: int, fraud_rate: Optional[float] = None) -> pd.DataFrame:
        """Generate synthetic insurance claims data."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before generating data")
        
        # Generate noise
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate synthetic data
        synthetic_data = self.generator.predict(noise, verbose=0)
        
        # Inverse transform to original scale
        synthetic_data = self.scaler.inverse_transform(synthetic_data)
        
        # Create DataFrame
        df_synthetic = pd.DataFrame(synthetic_data, columns=self.feature_names)
        
        # Ensure realistic constraints
        df_synthetic = self._apply_constraints(df_synthetic)
        
        # Generate fraud labels
        if fraud_rate is not None:
            df_synthetic['is_fraud'] = np.random.binomial(1, fraud_rate, n_samples)
        else:
            # Use realistic fraud probability based on features
            fraud_prob = self._estimate_fraud_probability(df_synthetic)
            df_synthetic['is_fraud'] = np.random.binomial(1, fraud_prob)
        
        return df_synthetic
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic constraints to generated data."""
        
        df = df.copy()
        
        # Age constraints
        if 'age' in df.columns:
            df['age'] = np.clip(df['age'], 18, 85).astype(int)
        
        # Vehicle age constraints
        if 'vehicle_age' in df.columns:
            df['vehicle_age'] = np.clip(df['vehicle_age'], 0, 30).astype(int)
        
        # Mileage constraints
        if 'annual_mileage' in df.columns:
            df['annual_mileage'] = np.clip(df['annual_mileage'], 1000, 100000).astype(int)
        
        # Violations constraints
        if 'driving_violations' in df.columns:
            df['driving_violations'] = np.clip(df['driving_violations'], 0, 20).astype(int)
        
        # Claim amount constraints
        if 'claim_amount' in df.columns:
            df['claim_amount'] = np.clip(df['claim_amount'], 100, 500000).astype(int)
        
        # Previous claims constraints
        if 'previous_claims' in df.columns:
            df['previous_claims'] = np.clip(df['previous_claims'], 0, 20).astype(int)
        
        # Credit score constraints
        if 'credit_score' in df.columns:
            df['credit_score'] = np.clip(df['credit_score'], 300, 850).astype(int)
        
        return df
    
    def _estimate_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate realistic fraud probability based on features."""
        
        # Simple heuristic based on risk factors
        fraud_prob = np.full(len(df), 0.1)  # Base rate
        
        if 'driving_violations' in df.columns:
            fraud_prob += 0.05 * (df['driving_violations'] > 2)
        
        if 'previous_claims' in df.columns:
            fraud_prob += 0.08 * (df['previous_claims'] > 2)
        
        if 'claim_amount' in df.columns:
            fraud_prob += 0.1 * (df['claim_amount'] > df['claim_amount'].quantile(0.8))
        
        if 'credit_score' in df.columns:
            fraud_prob += 0.05 * (df['credit_score'] < 500)
        
        if 'age' in df.columns:
            fraud_prob += 0.03 * (df['age'] < 25)
        
        return np.clip(fraud_prob, 0, 1)

class VariationalAutoencoder:
    """Simpler VAE approach for data augmentation."""
    
    def __init__(self, latent_dim: int = 20):
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.scaler = None
        self.feature_names = None
        
    def build_encoder(self, input_dim: int) -> keras.Model:
        """Build encoder network."""
        
        inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        
        z_mean = keras.layers.Dense(self.latent_dim)(x)
        z_log_sigma = keras.layers.Dense(self.latent_dim)(x)
        
        return keras.Model(inputs, [z_mean, z_log_sigma])
    
    def build_decoder(self, output_dim: int) -> keras.Model:
        """Build decoder network."""
        
        latent_inputs = keras.layers.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(64, activation='relu')(latent_inputs)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(output_dim, activation='linear')(x)
        
        return keras.Model(latent_inputs, outputs)
    
    def sampling(self, args):
        """Reparameterization trick."""
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon
    
    def train_simple_generator(self, df: pd.DataFrame, epochs: int = 100) -> Dict[str, Any]:
        """Train a simple noise-based generator as fallback."""
        
        # Prepare data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')
        
        self.feature_names = numeric_cols
        data = df[numeric_cols].values
        
        # Simple statistics-based generation
        self.feature_stats = {}
        for i, col in enumerate(numeric_cols):
            self.feature_stats[col] = {
                'mean': np.mean(data[:, i]),
                'std': np.std(data[:, i]),
                'min': np.min(data[:, i]),
                'max': np.max(data[:, i])
            }
        
        logger.info("Simple statistical generator trained")
        
        return {'method': 'statistical', 'features': len(numeric_cols)}
    
    def generate_simple_data(self, n_samples: int) -> pd.DataFrame:
        """Generate data using simple statistical approach."""
        
        synthetic_data = {}
        
        for col, stats in self.feature_stats.items():
            # Generate from normal distribution, then clip to realistic range
            samples = np.random.normal(stats['mean'], stats['std'], n_samples)
            samples = np.clip(samples, stats['min'], stats['max'])
            synthetic_data[col] = samples
        
        df_synthetic = pd.DataFrame(synthetic_data)
        
        # Apply constraints
        df_synthetic = self._apply_simple_constraints(df_synthetic)
        
        # Add fraud labels
        fraud_prob = 0.15  # Slightly higher than normal to create challenging examples
        df_synthetic['is_fraud'] = np.random.binomial(1, fraud_prob, n_samples)
        
        return df_synthetic
    
    def _apply_simple_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply simple constraints to generated data."""
        
        df = df.copy()
        
        # Round integer columns
        int_cols = ['age', 'vehicle_age', 'annual_mileage', 'driving_violations', 
                   'claim_amount', 'previous_claims', 'credit_score']
        
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        
        return df

if __name__ == "__main__":
    from data_loader import InsuranceDataLoader
    
    # Load data
    loader = InsuranceDataLoader()
    data = loader.load_data()
    
    print("Original data shape:", data.shape)
    print("Fraud rate:", data['is_fraud'].mean())
    
    # Try simple generator first
    generator = SyntheticDataGenerator()
    vae = VariationalAutoencoder()
    
    # Train simple generator
    vae.train_simple_generator(data)
    
    # Generate synthetic data
    synthetic_data = vae.generate_simple_data(2000)
    
    print("Synthetic data shape:", synthetic_data.shape)
    print("Synthetic fraud rate:", synthetic_data['is_fraud'].mean())
    
    # Compare distributions
    print("\nFeature comparison (mean):")
    for col in synthetic_data.columns:
        if col in data.columns and col != 'is_fraud':
            print(f"{col}: Original={data[col].mean():.2f}, Synthetic={synthetic_data[col].mean():.2f}")