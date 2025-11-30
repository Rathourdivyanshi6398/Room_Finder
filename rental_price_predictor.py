import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import joblib
import json

def extract_bhk(title):
    """Extract BHK number from property title."""
    match = re.search(r'(\d+)\s*BHK', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def clean_area(area_str):
    """Convert area string to float by removing 'sq.ft'."""
    try:
        return float(area_str.replace(' sq.ft', '').strip())
    except:
        return None

def load_and_preprocess_data():
    """Load and preprocess the rental property datasets from both cities."""
    try:
        # Load both datasets with explicit encoding
        print("Attempting to load Lucknow data...")
        lucknow_df = pd.read_csv('data/lucknow_data.csv', encoding='utf-8')
        print("Lucknow data loaded successfully")
        print(f"Lucknow data shape: {lucknow_df.shape}")
        
        print("\nAttempting to load Noida data...")
        noida_df = pd.read_csv('data/noida_data.csv', encoding='utf-8')
        print("Noida data loaded successfully")
        print(f"Noida data shape: {noida_df.shape}")
        
        # Standardize column names to lowercase
        lucknow_df.columns = lucknow_df.columns.str.lower()
        noida_df.columns = noida_df.columns.str.lower()
        
        # Combine the datasets
        df = pd.concat([lucknow_df, noida_df], ignore_index=True)
        
        print(f"\nTotal number of properties: {len(df)}")
        
        # Extract BHK from title
        df['bhk'] = df['title'].apply(extract_bhk)
        
        # Clean area column
        df['area_clean'] = df['area'].apply(clean_area)
        
        # Create BHK-based features
        df['bhk_area'] = df['bhk'] * df['area_clean']  # Interaction between BHK and area
        df['bhk_squared'] = df['bhk'] ** 2  # Non-linear BHK effect
        
        # Drop rows with missing values
        df = df.dropna(subset=['price', 'bhk', 'area_clean', 'city', 'location'])
        
        print(f"\nNumber of properties after cleaning: {len(df)}")
        
        # Encode categorical variables
        le_city = LabelEncoder()
        le_location = LabelEncoder()
        
        df['city_encoded'] = le_city.fit_transform(df['city'])
        df['location_encoded'] = le_location.fit_transform(df['location'])
        
        # Save location lists for frontend
        location_data = {
            'Lucknow': sorted(df[df['city'] == 'Lucknow']['location'].unique().tolist()),
            'Noida': sorted(df[df['city'] == 'Noida']['location'].unique().tolist())
        }
        with open('static/locations.json', 'w') as f:
            json.dump(location_data, f)
        
        # Save encoders
        joblib.dump(le_city, 'city_encoder.joblib')
        joblib.dump(le_location, 'location_encoder.joblib')
        
        return df
    except Exception as e:
        print(f"Error loading or preprocessing data: {str(e)}")
        raise

def train_model(df):
    """Train the Random Forest model."""
    # Prepare features and target
    X = df[['bhk', 'area_clean', 'city_encoded', 'location_encoded', 'bhk_area', 'bhk_squared']]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model with adjusted parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance Metrics:")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': ['BHK', 'Area', 'City', 'Location', 'BHK-Area Interaction', 'BHK Squared'],
        'importance': model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
    
    # Save model
    joblib.dump(model, 'model.joblib')
    
    return model, X_test, y_test

def predict_price(city, location, bhk, area, model, le_city, le_location):
    """Predict rental price for given features."""
    try:
        # Encode categorical variables
        city_encoded = le_city.transform([city])[0]
        location_encoded = le_location.transform([location])[0]
        
        # Create feature array with new features
        features = np.array([[
            bhk,
            area,
            city_encoded,
            location_encoded,
            bhk * area,  # BHK-Area interaction
            bhk ** 2    # BHK squared
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        return round(prediction, 2)
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Load the saved encoders
        le_city = joblib.load('city_encoder.joblib')
        le_location = joblib.load('location_encoder.joblib')
        
        # Train model
        model, X_test, y_test = train_model(df)
        
        print("\nModel training completed successfully!")
        print("Model and encoders have been saved to disk.")
        
        # Example predictions for both cities
        print("\nExample Predictions:")
        
        # Lucknow example
        lucknow_prediction = predict_price(
            city='Lucknow',
            location='Gomti Nagar Extension',
            bhk=3,
            area=1425,
            model=model,
            le_city=le_city,
            le_location=le_location
        )
        print(f"\nLucknow Example (3BHK in Gomti Nagar Extension):")
        print(f"Predicted Price: ₹{lucknow_prediction}")
        
        # Noida example
        noida_prediction = predict_price(
            city='Noida',
            location='Sector 137',
            bhk=2,
            area=1045,
            model=model,
            le_city=le_city,
            le_location=le_location
        )
        print(f"\nNoida Example (2BHK in Sector 137):")
        print(f"Predicted Price: ₹{noida_prediction}")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 