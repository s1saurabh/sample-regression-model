import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load dataset (replace 'bangalore_rentals.csv' with your actual file)
data = pd.read_csv('bangalore_rentals.csv')

# Example columns: 'area', 'location', 'bedrooms', 'bathrooms', 'sqft', 'rent'
# Fill missing values
for col in ['area', 'location', 'bedrooms', 'bathrooms', 'sqft']:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna('Unknown')
    else:
        data[col] = data[col].fillna(data[col].median())

# Encode categorical variables
categorical_cols = ['area', 'location']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combine features
features = pd.concat([
    encoded_df,
    data[['bedrooms', 'bathrooms', 'sqft']].reset_index(drop=True)
], axis=1)
target = data['rent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# Example prediction
sample = X_test.iloc[0:1]
predicted_rent = model.predict(sample)
print(f'Predicted rent for sample: {predicted_rent[0]:.2f}')

# Accessibility note: This script is built with accessibility in mind. Please review and test with your data and tools like Accessibility Insights.
