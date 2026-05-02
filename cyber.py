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
file_name = 'cybersecurity.csv'
file_path = os.path.join(data_dir, file_name)

def train_multiple_models(path):
    try:
        if not os.path.exists(path):
            print(f"Error: The file {path} was not found.")
            return None, None, None, None, None

        print(f"Loading data from: {path}...")
        df = pd.read_csv(path)
        
        # 2. Preprocessing
        cols_to_drop = ['id', 'ID', 'Timestamp', 'timestamp', 'User ID', 'Unnamed: 0']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        encoders = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
            
        X = df.iloc[:, :-1]  
        y = df.iloc[:, -1]   
        
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
            
        return trained_models, encoders, scaler, X.columns, df

    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None, None, None

def predict_from_dataset(models, scaler, df):
    # SCANNER LOGIC: Automatically find attack rows to help the user
    attack_indices = df[df.iloc[:, -1] == 1].index.tolist()
    
    print("\n" + "="*50)
    print("--- CYBER ATTACK SCANNER ---")
    if attack_indices:
        print(f"Found {len(attack_indices)} total attacks in this dataset.")
        print(f"Try these row indices for an ATTACK (1): {attack_indices[:10]}")
    else:
        print("No attacks (Label 1) found in the dataset.")
    print("="*50)

    print("\n--- Testing with Dataset Values ---")
    print(f"Available rows: {len(df)}")
    
    choice = input("Enter a row index or 'r' for a random row: ")
    
    try:
        if choice.lower() == 'r':
            sample_row = df.sample(n=1)
        else:
            idx = int(choice)
            sample_row = df.iloc[[idx]]
            
        actual_label = sample_row.iloc[0, -1]
        features = sample_row.iloc[:, :-1]
        
        input_scaled = scaler.transform(features)
        
        print(f"\n--- Results for Row {sample_row.index[0]} ---")
        print(f"Actual Label in CSV: {actual_label} (0=Normal, 1=Attack)")
        print("-" * 45)
        
        for name, model in models.items():
            prediction = model.predict(input_scaled)
            print(f"[{name}] Prediction: {prediction[0]}")
            
    except Exception as e:
        print(f"Invalid input or error: {e}")

if __name__ == "__main__":
    trained_suite, label_encoders, data_scaler, column_names, processed_df = train_multiple_models(file_path)
    
    if trained_suite:
        while True:
            predict_from_dataset(trained_suite, data_scaler, processed_df)
            
            user_choice = input("\nTry another row? (y/n): ").lower()
            if user_choice != 'y':
                print("Exiting program.")
                break