import pickle
import pandas as pd
from prophet import Prophet
import os

os.makedirs('models', exist_ok=True)

# Sample data (replace with your real data later)
data = {
    'ds': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'y': [100 + i % 30 + (i % 7) * 5 for i in range(365)]
}

df = pd.DataFrame(data)

products = {
    'milk': df.copy(),
    'newspaper': df.copy(),
}

for product_name, product_df in products.items():
    print(f"Training model for {product_name}...")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='IN')
    model.fit(product_df)
    
    with open(f'models/{product_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ {product_name} model saved!")

print("🎉 All models trained!")