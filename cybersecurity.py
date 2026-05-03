import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Configuration - Using relative path to avoid "No such file" errors
# This looks for the file in the same folder as this script
data_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'cybersecurity.csv'
file_path = os.path.join(data_dir, file_name)

def train_multiple_models(path):
    try:
        if not os.path.exists(path):
            print(f"Error: The file {path} was not found.")
            print("Current Directory Contents:", os.listdir(data_dir))
            return None, None, None, None, None

        print(f"Loading data from: {path}...")
        df = pd.read_csv(path)
        
        # --- IMPROVED BINARY FIX ---
        # Checks for multiple common 'Normal' labels to prevent the 1-class error
        target_col = df.columns[-1]
        df[target_col] = df[target_col].apply(
            lambda x: 0 if str(x).lower().strip() in ['0', 'normal', 'benign', 'safe'] else 1
        )
        
        # --- DATA VERIFICATION ---
        class_counts = df[target_col].value_counts()
        print("\nDataset Class Distribution:")
        print(class_counts)
        
        if len(class_counts) < 2:
            print("\nCRITICAL ERROR: Your dataset only contains one class.")
            print("Check your CSV labels. The models need both 'Normal' and 'Attack' data to train.")
            return None, None, None, None, None

        # 2. Preprocessing
        cols_to_drop = ['id', 'ID', 'Timestamp', 'timestamp', 'User ID', 'Unnamed: 0']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        encoders = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if col != target_col:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            
        X = df.iloc[:, :-1]  
        y = df.iloc[:, -1]   
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # --- STRATIFIED SPLIT ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
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

def predict_logic(models, scaler, feature_df, actual_label="Unknown"):
    input_scaled = scaler.transform(feature_df)
    print("\n" + "-"*30)
    print(f"Target Label (Truth): {actual_label}")
    print("-" * 30)
    for name, model in models.items():
        prediction = model.predict(input_scaled)
        print(f"[{name}] Prediction: {prediction[0]}")

def get_predictions(models, encoders, scaler, column_names, df):
    print("\n" + "="*50)
    print("HOW DO YOU WANT TO PREDICT?")
    print("[1] Pull a row from CSV (Automated Scanner)")
    print("[2] Enter values manually")
    mode = input("Select option (1 or 2): ")

    if mode == '1':
        # Select an attack row if possible for testing
        attack_indices = df[df.iloc[:, -1] == 1].index.tolist()
        if attack_indices:
            print(f"Found {len(attack_indices)} attacks. Try indices: {attack_indices[:10]}")
        
        choice = input(f"Enter row index (0-{len(df)-1}) or 'r' for random: ")
        try:
            if choice.lower() == 'r':
                sample_row = df.sample(n=1)
            else:
                sample_row = df.iloc[[int(choice)]]
            
            actual_val = sample_row.iloc[0, -1]
            features = sample_row.iloc[:, :-1]
            predict_logic(models, scaler, features, actual_val)
        except Exception as e:
            print(f"Invalid input: {e}")

    elif mode == '2':
        print("\n--- Manual Entry ---")
        custom_entry = {}
        for feature in column_names:
            val = input(f"Enter value for '{feature}': ")
            if feature in encoders:
                try:
                    custom_entry[feature] = encoders[feature].transform([str(val)])[0]
                except:
                    custom_entry[feature] = 0 
            else:
                try:
                    custom_entry[feature] = float(val)
                except:
                    custom_entry[feature] = 0.0
        
        manual_df = pd.DataFrame([custom_entry])
        predict_logic(models, scaler, manual_df, "User Provided")

if __name__ == "__main__":
    trained_suite, label_encoders, data_scaler, cols, processed_df = train_multiple_models(file_path)
    
    if trained_suite:
        while True:
            get_predictions(trained_suite, label_encoders, data_scaler, cols, processed_df)
            user_choice = input("\nRun another prediction? (y/n): ").lower()
            if user_choice != 'y':
                print("Exiting.")
                break