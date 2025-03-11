import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # To save/load model

# Load dataset
data = pd.read_csv("insurance.csv")

# Prepare data
X = data.drop(['PremiumPrice'], axis=1)
y = data['PremiumPrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gr = GradientBoostingRegressor()
gr.fit(X_train, y_train)

# Save the trained model
joblib.dump(gr, "premium_model.pkl")
print("Model saved successfully!")
