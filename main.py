import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("Credit_Card_Default.csv")

# Features and target
X = df.drop(columns=['default.payment.next.month', 'ID'], errors='ignore')
y = df['default.payment.next.month']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open("credit_default_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model saved to 'credit_default_model.pkl'")
