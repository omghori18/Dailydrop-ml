from flask import Flask, jsonify, request
from prophet import Prophet
import pandas as pd
import os
import pickle

app = Flask(__name__)

def train_model(product_name):
    os.makedirs('models', exist_ok=True)
    
    data = {
        'ds': pd.date_range(start='2023-01-01', periods=365, freq='D'),
        'y': [100 + i % 30 + (i % 7) * 5 for i in range(365)]
    }
    df = pd.DataFrame(data)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name='IN')
    model.fit(df)

    with open(f'models/{product_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def load_model(product_name):
    model_path = f'models/{product_name}_model.pkl'
    if not os.path.exists(model_path):
        # Train model if not exists
        return train_model(product_name)
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except:
        # If loading fails retrain
        return train_model(product_name)

@app.route('/')
def home():
    return jsonify({"message": "Daily Drop ML API is running! 🚀"})

@app.route('/predict')
def predict():
    product = request.args.get('product', 'milk')
    days = int(request.args.get('days', 30))

    model = load_model(product)

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