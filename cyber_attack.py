import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Configuration 
data_dir = r"C:\Users\ganga\OneDrive\Documents\nikit ml projects\cyber security attacks"
file_name ='cybersecurity.csv'
file_path = os.path.join(data_dir, file_name)

def train_multiple_models(path):
    try:
        if not os.path.exists(path):
            print(f"Error: The file {path} was not found.")
            return None, None, None, None

        print(f"Loading data from: {path}...")
        df = pd.read_csv(path)
        
        # 2. Preprocessing
        # Drop non-predictive columns that often cause low accuracy
        cols_to_drop = ['id', 'ID', 'Timestamp', 'timestamp', 'User ID', 'Unnamed: 0']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        encoders = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
            
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target (Last column)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 3. Initialize and Train Ensemble
        models = {
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            print(f"--- Training {name} ---")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"{name} Validation Accuracy: {acc*100:.2f}%")
            trained_models[name] = model
            
        return trained_models, encoders, scaler, X.columns

    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None, None

def predict_user_input(models, encoders, scaler, feature_names):
    print("\n--- Enter Custom Network Data Values ---")
    custom_entry = {}
    
    for feature in feature_names:
        val = input(f"Enter value for '{feature}': ")
        
        if feature in encoders:
            try:
                custom_entry[feature] = encoders[feature].transform([val])[0]
            except:
                custom_entry[feature] = 0
        else:
            try:
                custom_entry[feature] = float(val)
            except:
                custom_entry[feature] = 0.0

    # Scale the input
    df_input = pd.DataFrame([custom_entry])
    input_scaled = scaler.transform(df_input)
    
    print("\n>>> PREDICTION RESULTS (0=Normal, 1=Attack):")
    for name, model in models.items():
        prediction = model.predict(input_scaled)
        print(f"[{name}] Prediction: {prediction[0]}")

if __name__ == "__main__":
    # Ensure the script runs from the correct directory
    trained_suite, label_encoders, data_scaler, column_names = train_multiple_models(file_path)
    
    if trained_suite:
        while True:
            predict_user_input(trained_suite, label_encoders, data_scaler, column_names)
            user_choice = input("\nPredict another entry? (y/n): ").lower()
            if user_choice != 'y':
                print("Exiting program.")
                break