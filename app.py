from flask import Flask, jsonify, request
from prophet import Prophet
import pandas as pd
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta

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
    print(f"Fetching serviceEntries for provider: {provider_id}")
    entries = db.collection('serviceEntries')\
        .where('providerId', '==', provider_id)\
        .stream()

    records = []
    for entry in entries:
        data = entry.to_dict()
        records.append({
            'ds': data.get('date'),
            'y': float(data.get('quantity', 0))
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df['y'] = df['y'].clip(lower=0)
    df = df.groupby('ds')['y'].sum().reset_index()
    df = df.sort_values('ds')
    print(f"Total records found: {len(df)}")
    return df

def fetch_customer_data(provider_id, service_type):
    print(f"Fetching customers for: {service_type}")
    customers = db.collection('customers')\
        .where('providerId', '==', provider_id)\
        .where('status', '==', 'ACTIVE')\
        .stream()

    total_daily = 0
    customer_count = 0

    for customer in customers:
        data = customer.to_dict()
        if data.get('serviceType', '').lower() == service_type.lower():
            qty = float(data.get('defaultQuantity', 0))
            total_daily += qty
            customer_count += 1

    print(f"Active customers: {customer_count}, Daily total: {total_daily}")
    return total_daily, customer_count

def get_sample_data():
    data = {
        'ds': pd.date_range(start='2024-01-01', periods=60, freq='D'),
        'y': [abs(10 + i % 7 * 2) for i in range(60)]
    }
    return pd.DataFrame(data)

def predict_with_prophet(df, days):
    # Make sure all values are positive
    df['y'] = df['y'].abs()

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=len(df) >= 14,
        daily_seasonality=False,
        changepoint_prior_scale=0.01
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    result['ds'] = result['ds'].astype(str)

    # Force all positive
    result['yhat'] = result['yhat'].abs()
    result['yhat_lower'] = result['yhat_lower'].abs()
    result['yhat_upper'] = result['yhat_upper'].abs()

    return result

@app.route('/')
def home():
    return jsonify({"message": "Daily Drop ML API is running! 🚀"})

@app.route('/predict')
def predict():
    provider_id = request.args.get('providerId', '')
    product = request.args.get('product', 'Milk')
    days = int(request.args.get('days', 30))

    if not provider_id:
        return jsonify({"error": "providerId is required"}), 400

    # Step 1 — Try serviceEntries data
    df = fetch_firestore_data(provider_id)

    # Step 2 — If not enough data, build from customer defaultQuantity
    if df is None or len(df) < 5:
        print("Not enough serviceEntries, using customer defaultQuantity...")
        total_daily, customer_count = fetch_customer_data(provider_id, product)

        if total_daily > 0:
            # Build fake history from defaultQuantity
            dates = pd.date_range(
                start=datetime.today() - timedelta(days=60),
                periods=60,
                freq='D'
            )
            df = pd.DataFrame({
                'ds': dates,
                'y': [total_daily] * 60
            })
        else:
            print("No customer data either, using sample data...")
            df = get_sample_data()

    # Step 3 — Predict with Prophet
    result = predict_with_prophet(df, days)

    total_predicted = round(result['yhat'].sum(), 2)

    return jsonify({
        "product": product,
        "provider_id": provider_id,
        "days": days,
        "total_predicted": total_predicted,
        "average_daily": round(total_predicted / days, 2),
        "daily": result.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)