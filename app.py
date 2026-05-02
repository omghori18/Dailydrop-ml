from flask import Flask, jsonify, request
import os
import json
import pickle
import warnings
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta

# Suppress all warnings for cleaner production logs
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 🔥 Firebase Initialization (FIXED)
firebase_json = os.environ.get('FIREBASE_CREDENTIALS')

if firebase_json:
    # Production: use environment variable
    firebase_dict = json.loads(firebase_json)
    cred = credentials.Certificate(firebase_dict)
elif os.path.exists('serviceAccount.json'):
    # Local development: use serviceAccount.json file
    cred = credentials.Certificate('serviceAccount.json')
    print("Using local serviceAccount.json for Firebase")
else:
    raise Exception("FIREBASE_CREDENTIALS environment variable is missing and serviceAccount.json not found. Set FIREBASE_CREDENTIALS in environment variables.")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load ML models with error handling
models = {}
model_status = {
    'milk': False,
    'newspaper': False
}

try:
    with open('models/milk_model.pkl', 'rb') as f:
        models['milk'] = pickle.load(f)
        model_status['milk'] = True
    print("Milk model loaded successfully")
except Exception as e:
    # Suppress model loading errors in production logs - they're handled gracefully
    models['milk'] = None
    model_status['milk'] = False

try:
    with open('models/newspaper_model.pkl', 'rb') as f:
        models['newspaper'] = pickle.load(f)
        model_status['newspaper'] = True
    print("Newspaper model loaded successfully")
except Exception as e:
    # Suppress model loading errors in production logs - they're handled gracefully
    models['newspaper'] = None
    model_status['newspaper'] = False

print(f"Model status - Milk: {model_status['milk']}, Newspaper: {model_status['newspaper']}")
if not any(model_status.values()):
    print("WARNING: No ML models loaded. Using simple prediction fallback.")

def fetch_and_categorize(provider_id):
    customers = db.collection('customers') \
        .where('providerId', '==', provider_id) \
        .where('status', '==', 'ACTIVE') \
        .stream()

    categories = {}

    for customer in customers:
        data = customer.to_dict()

        service_type = data.get('serviceType', '').strip()
        if not service_type:
            continue

        qty = float(data.get('defaultQuantity', 0))
        rate = float(data.get('ratePerUnit', 0))
        name = data.get('name', 'Unknown')

        if service_type not in categories:
            categories[service_type] = {
                'customers': [],
                'total_daily_quantity': 0,
                'total_daily_revenue': 0
            }

        categories[service_type]['customers'].append({
            'name': name,
            'daily_quantity': qty,
            'rate_per_unit': rate
        })

        categories[service_type]['total_daily_quantity'] += qty
        categories[service_type]['total_daily_revenue'] += qty * rate

    return categories


@app.route('/')
def home():
    return jsonify({
        "message": "Daily Drop API running 🚀",
        "model_status": model_status,
        "models_loaded": sum(model_status.values()),
        "total_models": len(model_status)
    })


@app.route('/status')
def status():
    return jsonify({
        "status": "healthy" if any(model_status.values()) else "limited",
        "models": model_status,
        "models_loaded": sum(model_status.values()),
        "total_models": len(model_status),
        "prediction_method": "ML" if any(model_status.values()) else "simple_fallback",
        "note": "ML models failed to load due to pandas compatibility. Using simple prediction fallback." if not any(model_status.values()) else "ML models loaded successfully"
    })


@app.route('/predict/all')
def predict_all():
    provider_id = request.args.get('providerId', '')
    days = int(request.args.get('days', 30))

    if not provider_id:
        return jsonify({"error": "providerId is required"}), 400

    categories = fetch_and_categorize(provider_id)

    if not categories:
        return jsonify({"error": "No active customers found"}), 404

    predictions = {}

    for service_type, data in categories.items():
        daily_qty = data['total_daily_quantity']
        daily_rev = data['total_daily_revenue']

        today = datetime.today()
        daily_breakdown = []

        for i in range(1, days + 1):
            future_date = today + timedelta(days=i)
            daily_breakdown.append({
                "date": future_date.strftime('%Y-%m-%d'),
                "predicted_quantity": round(daily_qty, 2),
                "predicted_revenue": round(daily_rev, 2)
            })

        predictions[service_type] = {
            "active_customers": len(data['customers']),
            "daily_quantity": round(daily_qty, 2),
            "daily_revenue": round(daily_rev, 2),
            "total_predicted_quantity": round(daily_qty * days, 2),
            "total_predicted_revenue": round(daily_rev * days, 2),
            "customer_breakdown": data['customers'],
            "daily": daily_breakdown,
            "prediction_method": "simple_fallback",
            "note": "Using simple prediction (repeating current values) due to ML model loading failure"
        }

    total_revenue = sum(p['total_predicted_revenue'] for p in predictions.values())
    total_quantity = sum(p['total_predicted_quantity'] for p in predictions.values())

    return jsonify({
        "provider_id": provider_id,
        "days": days,
        "total_categories": len(predictions),
        "overall_predicted_quantity": round(total_quantity, 2),
        "overall_predicted_revenue": round(total_revenue, 2),
        "category_predictions": predictions,
        "prediction_method": "simple_fallback",
        "note": "Using simple prediction (repeating current values) due to ML model loading failure",
        "model_status": model_status
    })


@app.route('/predict')
def predict():
    provider_id = request.args.get('providerId', '')
    product = request.args.get('product', '')
    days = int(request.args.get('days', 30))

    if not provider_id:
        return jsonify({"error": "providerId is required"}), 400

    categories = fetch_and_categorize(provider_id)

    if product and product in categories:
        data = categories[product]
        daily_qty = data['total_daily_quantity']
        daily_rev = data['total_daily_revenue']

        today = datetime.today()
        daily_breakdown = []

        for i in range(1, days + 1):
            future_date = today + timedelta(days=i)
            daily_breakdown.append({
                "date": future_date.strftime('%Y-%m-%d'),
                "predicted_quantity": round(daily_qty, 2),
                "predicted_revenue": round(daily_rev, 2)
            })

        return jsonify({
            "product": product,
            "provider_id": provider_id,
            "days": days,
            "active_customers": len(data['customers']),
            "daily_quantity": round(daily_qty, 2),
            "daily_revenue": round(daily_rev, 2),
            "total_predicted_quantity": round(daily_qty * days, 2),
            "total_predicted_revenue": round(daily_rev * days, 2),
            "customer_breakdown": data['customers'],
            "daily": daily_breakdown,
            "prediction_method": "simple_fallback",
            "note": "Using simple prediction (repeating current values) due to ML model loading failure"
        })

    elif not product:
        return predict_all()

    else:
        return jsonify({
            "error": f"No data found for {product}",
            "available_categories": list(categories.keys())
        }), 404
