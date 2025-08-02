"""
Data preprocessing module for board game recommendation system.
Handles cleaning, normalization, and feature engineering of board game data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os


class BoardGameDataPreprocessor:
    """Preprocesses board game data for recommendation system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.mechanics_binarizer = MultiLabelBinarizer()
        self.domains_binarizer = MultiLabelBinarizer()
        self.processed_data = None
        self.original_data = None
        
    def load_data(self, filepath):
        """Load and perform initial cleaning of the dataset."""
        print("Loading dataset...")
        
        # Load CSV with proper encoding and separator
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        
        print(f"Loaded {len(df)} games")
        print(f"Columns: {list(df.columns)}")
        
        # Basic data cleaning
        df = self._clean_basic_data(df)
        
        self.original_data = df.copy()
        return df
    
    def _clean_basic_data(self, df):
        """Perform basic data cleaning."""
        print("Performing basic data cleaning...")
        
        # Replace commas with dots in numeric columns (European format)
        numeric_cols = ['Rating Average', 'Complexity Average']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Remove rows with missing essential data
        essential_cols = ['Name', 'Min Players', 'Max Players', 'Play Time']
        df = df.dropna(subset=essential_cols)
        
        # Fill missing values
        df['Rating Average'] = df['Rating Average'].fillna(df['Rating Average'].median())
        df['Complexity Average'] = df['Complexity Average'].fillna(df['Complexity Average'].median())
        df['Min Age'] = df['Min Age'].fillna(df['Min Age'].median())
        df['Year Published'] = df['Year Published'].fillna(df['Year Published'].median())
        
        # Clean mechanics and domains
        df['Mechanics'] = df['Mechanics'].fillna('')
        df['Domains'] = df['Domains'].fillna('')
        
        print(f"After cleaning: {len(df)} games remaining")
        return df
    
    def preprocess_features(self, df):
        """Preprocess and engineer features for recommendation."""
        print("Preprocessing features...")
        
        processed_df = df.copy()
        
        # 1. Normalize numerical features
        numerical_features = ['Play Time', 'Rating Average', 'Complexity Average', 'Min Age']
        processed_df[numerical_features] = self.scaler.fit_transform(processed_df[numerical_features])
        
        # 2. Create player count features
        processed_df['Player_Range'] = processed_df['Max Players'] - processed_df['Min Players']
        processed_df['Supports_2_Players'] = (processed_df['Min Players'] <= 2) & (processed_df['Max Players'] >= 2)
        processed_df['Supports_3_Players'] = (processed_df['Min Players'] <= 3) & (processed_df['Max Players'] >= 3)
        processed_df['Supports_4_Players'] = (processed_df['Min Players'] <= 4) & (processed_df['Max Players'] >= 4)
        processed_df['Supports_5_Plus'] = processed_df['Max Players'] >= 5
        
        # 3. Create time categories
        original_playtime = df['Play Time']  # Use original non-normalized values
        processed_df['Quick_Game'] = (original_playtime <= 30).astype(int)
        processed_df['Medium_Game'] = ((original_playtime > 30) & (original_playtime <= 90)).astype(int)
        processed_df['Long_Game'] = (original_playtime > 90).astype(int)
        
        # 4. Create complexity categories
        original_complexity = df['Complexity Average']  # Use original non-normalized values
        processed_df['Simple_Game'] = (original_complexity <= 2.0).astype(int)
        processed_df['Medium_Complexity'] = ((original_complexity > 2.0) & (original_complexity <= 3.5)).astype(int)
        processed_df['Complex_Game'] = (original_complexity > 3.5).astype(int)
        
        # 5. One-hot encode mechanics
        mechanics_list = self._parse_list_column(df['Mechanics'])
        mechanics_encoded = self.mechanics_binarizer.fit_transform(mechanics_list)
        mechanics_df = pd.DataFrame(
            mechanics_encoded, 
            columns=[f'Mechanic_{name}' for name in self.mechanics_binarizer.classes_],
            index=processed_df.index
        )
        
        # 6. One-hot encode domains
        domains_list = self._parse_list_column(df['Domains'])
        domains_encoded = self.domains_binarizer.fit_transform(domains_list)
        domains_df = pd.DataFrame(
            domains_encoded,
            columns=[f'Domain_{name}' for name in self.domains_binarizer.classes_],
            index=processed_df.index
        )
        
        # Combine all features
        final_df = pd.concat([processed_df, mechanics_df, domains_df], axis=1)
        
        self.processed_data = final_df
        print(f"Final feature count: {len(final_df.columns)}")
        return final_df
    
    def _parse_list_column(self, series):
        """Parse comma-separated list columns."""
        parsed_lists = []
        for item in series:
            if pd.isna(item) or item == '':
                parsed_lists.append([])
            else:
                # Split by comma and clean whitespace
                items = [x.strip() for x in str(item).split(',')]
                parsed_lists.append(items)
        return parsed_lists
    
    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor for later use."""
        preprocessor_data = {
            'scaler': self.scaler,
            'mechanics_binarizer': self.mechanics_binarizer,
            'domains_binarizer': self.domains_binarizer
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load a pre-fitted preprocessor."""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.mechanics_binarizer = preprocessor_data['mechanics_binarizer']
        self.domains_binarizer = preprocessor_data['domains_binarizer']
        print(f"Preprocessor loaded from {filepath}")
    
    def get_feature_names(self):
        """Get all feature names after preprocessing."""
        if self.processed_data is None:
            return None
        return list(self.processed_data.columns)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = BoardGameDataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data('../data/bgg_dataset.csv')
    processed_df = preprocessor.preprocess_features(df)
    
    # Save preprocessor
    preprocessor.save_preprocessor('../models/preprocessor.pkl')
    
    # Save processed data
    processed_df.to_csv('../data/processed_games.csv', index=False)
    
    print("Data preprocessing completed!")
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Sample columns: {processed_df.columns[:10].tolist()}")