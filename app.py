from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import math
import joblib
import numpy as np
from rental_price_predictor import extract_bhk, clean_area
import re
import os

app = Flask(__name__)
CORS(app)

CSV_FILE_PATH = 'final_lkn_out.csv'

# Initialize DataFrame
df = pd.DataFrame()

try:
    # Load and clean data
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"✅ Successfully loaded {len(df)} rows from {CSV_FILE_PATH}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Clean numeric columns
    numeric_columns = {
        'price': {
            'cleaners': [
                lambda x: str(x).replace('₹', '').replace(',', '').strip(),
                pd.to_numeric
            ],
            'na_action': 'drop'
        }
    }
    
    # Clean text columns
    text_columns = {
        'subtitle': {'default': ''},
        'title': {'default': 'No Title'},
        'area': {'default': 'Area Not Specified'},
        'property_details': {'default': 'No Details Available'},
        'city': {'default': 'Lucknow'},
        'location': {'default': ''}  # Add location column cleaning
    }
    
    # Process numeric columns
    for col, config in numeric_columns.items():
        if col in df.columns:
            for cleaner in config['cleaners']:
                df[col] = df[col].apply(cleaner)
            if config.get('na_action') == 'drop':
                initial_count = len(df)
                df.dropna(subset=[col], inplace=True)
                print(f"♻️  Cleaned '{col}': Removed {initial_count - len(df)} rows")
    
    # Process text columns
    for col, config in text_columns.items():
        if col in df.columns:
            df[col] = df[col].fillna(config['default']).astype(str).str.strip()
    
    print(f"🧹 Data cleaning complete. Final row count: {len(df)}")

except FileNotFoundError:
    print(f"❌ Critical error: File not found at {CSV_FILE_PATH}")
except Exception as e:
    print(f"🚨 Error during data loading: {str(e)}")
detailed_df = pd.read_csv('final_lkn_in.csv')
@app.route('/property-details')
def get_property_details():
    image_url = request.args.get('first_image_url')
    if not image_url:
        return jsonify({"error": "Missing first_image_url parameter"}), 400
    
    # Handle NaN values in the DataFrame
    property_details = detailed_df[detailed_df['first_image_url'] == image_url]\
        .fillna('').to_dict(orient='records')

    if not property_details:
        return jsonify({"error": "Property not found"}), 404

    # Convert numpy NaN to None explicitly
    cleaned_details = {k: v if pd.notna(v) else None for k, v in property_details[0].items()}
    return jsonify(cleaned_details)

@app.route('/search', methods=['GET'])
def search_rooms():
    """Handle room search requests with filtering and pagination"""
    try:
        # Get and validate parameters
        params = {
            'location': re.sub(r'\s+', ' ', request.args.get('location', '').strip()).lower(),
            'budget': request.args.get('budget', '').strip(),
            'page': max(1, request.args.get('page', 1, type=int)),
            'limit': max(1, request.args.get('limit', 12, type=int))
        }

        print(f"\n🔍 New search request:")
        print(f"   - Location: {params['location']}")
        print(f"   - Max budget: {params['budget']}")
        print(f"   - Page: {params['page']}")
        print(f"   - Items per page: {params['limit']}")
        print(f"   Initial dataset size: {len(df)} rows")

        if df.empty:
            return jsonify({
                'results': [],
                'pagination': {
                    'total_results': 0,
                    'total_pages': 0,
                    'current_page': params['page'],
                    'page_size': params['limit']
                }
            })

        # Apply filters to the entire dataset
        filtered_df = df.copy()
        
        # Location filter (search title, subtitle, and Location columns)
        if params['location']:
            location_query = params['location']
            mask = (
                filtered_df['subtitle'].str.lower().str.contains(location_query, na=False) |
                filtered_df['title'].str.lower().str.contains(location_query, na=False) |
                filtered_df['Location'].str.lower().str.contains(location_query, na=False)  # Added Location column
            )
            filtered_df = filtered_df[mask]
            print(f"📍 Location filter applied: {len(filtered_df)} rows remaining")

        # Budget filter with 10% buffer
        if params['budget']:
            try:
                max_budget = float(params['budget'])
                if max_budget <= 0:
                    raise ValueError("Budget must be greater than 0")
                
                # Apply 10% upper buffer
                filtered_df = filtered_df[filtered_df['price'] <= max_budget * 1.10]
                print(f"💰 Budget filter applied (≤{max_budget*1.10:.2f}): {len(filtered_df)} rows remaining")
            except ValueError as e:
                print(f"⚠️  Invalid budget value: {str(e)} - skipping budget filter")

        # Pagination calculations
        total_results = len(filtered_df)
        total_pages = math.ceil(total_results / params['limit']) if params['limit'] > 0 else 0
        offset = (params['page'] - 1) * params['limit']
        
        # Get paginated results
        paginated_df = filtered_df.iloc[offset:offset + params['limit']]
        results = paginated_df.to_dict(orient='records')
        
        print(f"📊 Returning {len(results)} results (page {params['page']}/{total_pages})")
        print(f"   Total matching results: {total_results}")
        
        return jsonify({
            'results': results,
            'pagination': {
                'total_results': total_results,
                'total_pages': total_pages,
                'current_page': params['page'],
                'page_size': params['limit']
            }
        })
        
    except Exception as e:
        print(f"🚨 Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

model = joblib.load('model.joblib')
city_encoder = joblib.load('city_encoder.joblib')
location_encoder = joblib.load('location_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        city = request.form.get('city')
        location = request.form.get('location')
        bhk = int(request.form.get('bhk'))
        area = float(request.form.get('area'))

        # Validate inputs
        if not all([city, location, bhk, area]):
            return jsonify({
                'success': False,
                'error': 'All fields are required'
            })

        if bhk < 1:
            return jsonify({
                'success': False,
                'error': 'BHK must be at least 1'
            })

        if area <= 0:
            return jsonify({
                'success': False,
                'error': 'Area must be greater than 0'
            })

        # Prepare features
        features = np.array([[
            bhk,
            area,
            city_encoder.transform([city])[0],
            location_encoder.transform([location])[0],
            bhk * area,  # BHK-Area interaction
            bhk ** 2    # BHK squared
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format the prediction as currency
        formatted_prediction = f"₹{prediction:,.2f}"

        return jsonify({
            'success': True,
            'prediction': formatted_prediction,
            'details': {
                'city': city,
                'location': location,
                'bhk': bhk,
                'area': area
            }
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your request'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)