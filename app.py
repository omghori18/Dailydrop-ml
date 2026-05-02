from flask import Flask, jsonify, request
import os
import json
import pickle
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

# Load ML models with error handling
models = {}
try:
    with open('models/milk_model.pkl', 'rb') as f:
        models['milk'] = pickle.load(f)
    print("Milk model loaded successfully")
except Exception as e:
    print(f"Error loading milk model: {e}")
    models['milk'] = None

try:
    with open('models/newspaper_model.pkl', 'rb') as f:
        models['newspaper'] = pickle.load(f)
    print("Newspaper model loaded successfully")
except Exception as e:
    print(f"Error loading newspaper model: {e}")
    models['newspaper'] = None

def fetch_and_categorize(provider_id):
    print(f"Fetching all active customers for provider: {provider_id}")

    customers = db.collection('customers')\
        .where('providerId', '==', provider_id)\
        .where('status', '==', 'ACTIVE')\
        .stream()

    # Category wise data
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
    return jsonify({"message": "Daily Drop ML API is running! 🚀"})

@app.route('/predict/all')
def predict_all():
    provider_id = request.args.get('providerId', '')
    days = int(request.args.get('days', 30))

    if not provider_id:
        return jsonify({"error": "providerId is required"}), 400

    # Fetch and categorize all customers
    categories = fetch_and_categorize(provider_id)

    if not categories:
        return jsonify({
            "error": "No active customers found for this provider"
        }), 404

    # Build predictions category wise
    predictions = {}

    for service_type, data in categories.items():
        daily_qty = data['total_daily_quantity']
        daily_rev = data['total_daily_revenue']

        # Generate daily breakdown
        daily_breakdown = []
        today = datetime.today()
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
            "daily": daily_breakdown
        }

    # Overall summary
    total_revenue = sum(
        p['total_predicted_revenue'] for p in predictions.values()
    )
    total_quantity = sum(
        p['total_predicted_quantity'] for p in predictions.values()
    )

    return jsonify({
        "provider_id": provider_id,
        "days": days,
        "total_categories": len(predictions),
        "overall_predicted_quantity": round(total_quantity, 2),
        "overall_predicted_revenue": round(total_revenue, 2),
        "category_predictions": predictions
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

        daily_breakdown = []
        today = datetime.today()
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
            "daily": daily_breakdown
        })

    elif not product:
        # Return all categories if no product specified
        return predict_all()

    else:
        return jsonify({
            "error": f"No active customers found for {product}",
            "available_categories": list(categories.keys())
        }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)