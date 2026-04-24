# cybre-attack-security

Network Intrusion Detection using Machine Learning Ensemble
This project implements a multi-model machine learning pipeline to detect cyber security attacks based on network traffic data. It utilizes an ensemble approach, comparing XGBoost, Random Forest, and Support Vector Machines (SVM) to classify network activity as either normal or an attack.

🚀 Overview
The system processes raw network logs, performs automated feature encoding and scaling, and trains three distinct classifiers. It then provides an interactive CLI (Command Line Interface) for users to input custom network packet data and receive real-time predictions from all three models.

📊 Model Performance
Based on the initial training runs, the models achieved the following validation accuracies:

XGBoost: 98.60%

Random Forest: 98.55%

SVM: 98.30%

🛠️ Requirements
To run this project, you need Python installed along with the following libraries:

pandas

numpy

scikit-learn

xgboost

You can install the dependencies using:

Bash
pip install pandas numpy scikit-learn xgboost

📂 Project Structure
cyber_attack.py: The main Python script containing the training and prediction logic.

cybersecurity.csv: The dataset containing network traffic features and labels (Ensure this is in the project directory).

⚙️ Execution Steps
1. Setup the Environment
Clone this repository to your local machine:

Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Prepare the Data
Ensure your dataset (cybersecurity.csv) is placed in the correct directory. If your path differs from the hardcoded path in the script, update the data_dir variable in the code:

Python
data_dir = r"./" # Current directory

3. Run the Program
Execute the script using the terminal:

Bash
python cyber_attack.py

4. Training Phase
The script will automatically:

Load and clean the data (dropping non-predictive columns like IDs and Timestamps).

Encode categorical variables.

Scale features using StandardScaler.

Train and evaluate the three models.

5. Interactive Prediction
Once training is complete, the program will prompt you to enter values for features like src_ip, dst_ip, protocol, and bytes_sent.

0 = Normal Traffic

1 = Cyber Attack/Intrusion

📝 Features Used for Prediction
The models analyze various network attributes, including:

Source/Destination IPs & Ports

Protocol Types

Traffic Volume (Bytes sent/received)

User Agent & URL data

Internal vs External traffic flags

