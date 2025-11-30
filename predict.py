import joblib
import numpy as np

def load_models():
    """Load the saved model and encoders."""
    try:
        model = joblib.load('rental_price_model.joblib')
        city_encoder = joblib.load('city_encoder.joblib')
        location_encoder = joblib.load('location_encoder.joblib')
        return model, city_encoder, location_encoder
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

def get_user_input():
    """Get property details from user."""
    print("\nEnter property details:")
    print("------------------------")
    
    # Get city
    while True:
        city = input("Enter city (Lucknow/Noida): ").strip()
        if city in ['Lucknow', 'Noida']:
            break
        print("Invalid city. Please enter either Lucknow or Noida.")
    
    # Get location
    location = input("Enter location (e.g., 'Gomti Nagar Extension' for Lucknow or 'Sector 137' for Noida): ").strip()
    
    # Get BHK
    while True:
        try:
            bhk = int(input("Enter number of BHK: "))
            if bhk > 0:
                break
            print("BHK must be greater than 0.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get area
    while True:
        try:
            area = float(input("Enter area in sq.ft: "))
            if area > 0:
                break
            print("Area must be greater than 0.")
        except ValueError:
            print("Please enter a valid number.")
    
    return city, location, bhk, area

def predict_rent():
    """Main function to predict rent."""
    # Load models
    model, city_encoder, location_encoder = load_models()
    if model is None:
        return
    
    try:
        while True:
            # Get input from user
            city, location, bhk, area = get_user_input()
            
            # Encode categorical variables
            try:
                city_encoded = city_encoder.transform([city])[0]
                location_encoded = location_encoder.transform([location])[0]
            except ValueError:
                print(f"\nError: Location '{location}' not found in training data.")
                print("Please try a different location from the training dataset.")
                continue
            
            # Make prediction
            features = np.array([[bhk, area, city_encoded, location_encoded]])
            prediction = model.predict(features)[0]
            
            # Display result
            print(f"\nPredicted Rent for {bhk} BHK in {location}, {city}:")
            print(f"Area: {area} sq.ft")
            print(f"Estimated Rent: ₹{prediction:,.2f}")
            
            # Ask if user wants to make another prediction
            again = input("\nWould you like to make another prediction? (yes/no): ").lower()
            if again != 'yes':
                break
        
        print("\nThank you for using the Rental Price Predictor!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Welcome to Rental Price Predictor!")
    predict_rent() 