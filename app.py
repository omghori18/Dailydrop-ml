from flask import Flask, jsonify, request
from prophet import Prophet
import pandas as pd
import os
import pickle
import json
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Initialize Firebase
firebase_json = os.environ.get('FIREBASE_CREDENTIALS')
if firebase_json:
    firebase_dict = json.loads(firebase_json)
    cred = credentials.Certificate(firebase_dict)
else:
    cred = credentials.Certificate('serviceAccount.json')

firebase_admin.initialize_app(cred)
db = firestore.client()

def fetch_firestore_data(provider_id):
    print(f"Fetching data for provider: {provider_id}")
    entries = db.collection('serviceEntries')\
        .where('providerId', '==', provider_id)\
        .stream()

    records = []
    for entry in entries:
        data = entry.to_dict()
        records.append({
            'ds': data.get('date'),
            'y': data.get('quantity', 0)
        })

    df = pd.DataFrame(records)
    if df.empty:
        return None

    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df = df.groupby('ds')['y'].sum().reset_index()
    df = df.sort_values('ds')
    return df

def get_sample_data():
    data = {
        'ds': pd.date_range(start='2023-01-01', periods=365, freq='D'),
        'y': [100 + i % 30 + (i % 7) * 5 for i in range(365)]
    }
    return pd.DataFrame(data)

def train_model(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='IN')
    model.fit(df)
    return model

@app.route('/')
def home():
    return jsonify({"message": "Daily Drop ML API is running! 🚀"})

@app.route('/predict')
def predict():
    provider_id = request.args.get('providerId', '')
    product = request.args.get('product', 'milk')
    days = int(request.args.get('days', 30))

    # Try real Firestore data first
    df = None
    if provider_id:
        df = fetch_firestore_data(provider_id)

    # Fallback to sample data
    if df is None or len(df) < 10:
        print("Using sample data...")
        df = get_sample_data()

    model = train_model(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    result['ds'] = result['ds'].astype(str)

    return jsonify({
        "product": product,
        "provider_id": provider_id,
        "days": days,
        "total_predicted": round(result['yhat'].sum(), 2),
        "daily": result.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)