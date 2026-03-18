from flask import Flask, jsonify, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

def load_model(product_name):
    model_path = f'models/{product_name}_model.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@app.route('/')
def home():
    return jsonify({"message": "Daily Drop ML API is running! 🚀"})

@app.route('/predict')
def predict():
    product = request.args.get('product', 'milk')
    days = int(request.args.get('days', 30))

    model = load_model(product)
    if not model:
        return jsonify({"error": f"No model found for {product}"}), 404

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    result['ds'] = result['ds'].astype(str)

    return jsonify({
        "product": product,
        "days": days,
        "total_predicted": round(result['yhat'].sum(), 2),
        "daily": result.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



