from flask import Flask, jsonify, request
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

def fetch_customer_data(provider_id, service_type):
    print(f"Fetching customers for: {service_type}")
    customers = db.collection('customers')\
        .where('providerId', '==', provider_id)\
        .where('status', '==', 'ACTIVE')\
        .stream()

    total_daily = 0
    customer_count = 0
    customer_list = []

    for customer in customers:
        data = customer.to_dict()
        if data.get('serviceType', '').lower() == service_type.lower():
            qty = float(data.get('defaultQuantity', 0))
            rate = float(data.get('ratePerUnit', 0))
            total_daily += qty
            customer_count += 1
            customer_list.append({
                "name": data.get('name', 'Unknown'),
                "daily_quantity": qty,
                "rate_per_unit": rate,
                "monthly_amount": round(qty * rate * 30, 2)
            })

    return total_daily, customer_count, customer_list

def fetch_service_entries(provider_id):
    print(f"Fetching service entries...")
    entries = db.collection('serviceEntries')\
        .where('providerId', '==', provider_id)\
        .stream()

    records = []
    for entry in entries:
        data = entry.to_dict()
        qty = float(data.get('quantity', 0))
        if qty > 0:
            records.append(qty)

    return records

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

    # Get customer default quantities
    total_daily, customer_count, customer_list = fetch_customer_data(
        provider_id, product
    )

    # Get historical average from service entries
    historical = fetch_service_entries(provider_id)
    if historical:
        avg_historical = sum(historical) / len(historical)
    else:
        avg_historical = total_daily

    # Use best available data
    if total_daily > 0:
        daily_prediction = total_daily
        source = "customer_default_quantity"
    elif avg_historical > 0:
        daily_prediction = avg_historical
        source = "historical_average"
    else:
        daily_prediction = 10
        source = "fallback"

    # Generate daily predictions
    daily_breakdown = []
    today = datetime.today()
    for i in range(1, days + 1):
        future_date = today + timedelta(days=i)
        daily_breakdown.append({
            "ds": future_date.strftime('%Y-%m-%d'),
            "predicted_quantity": round(daily_prediction, 2)
        })

    total_predicted = round(daily_prediction * days, 2)
    total_revenue = round(
        sum(c['monthly_amount'] for c in customer_list), 2
    ) if customer_list else 0

    return jsonify({
        "product": product,
        "provider_id": provider_id,
        "days": days,
        "prediction_source": source,
        "daily_quantity": round(daily_prediction, 2),
        "total_predicted_quantity": total_predicted,
        "estimated_revenue": total_revenue,
        "active_customers": customer_count,
        "customer_breakdown": customer_list,
        "daily": daily_breakdown
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)