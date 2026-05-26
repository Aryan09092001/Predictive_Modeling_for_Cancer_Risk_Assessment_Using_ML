# 🩺 Cancer Risk Prediction using Machine Learning

A multi-class classification project that predicts cancer risk levels (**Low**, **Medium**, **High**) from patient lifestyle, demographic, and clinical features. The project includes complete data analysis, model development, hyperparameter tuning, and an interactive Streamlit web app for real-time predictions.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App Locally](#-running-the-app-locally)
- [Deployment on AWS EC2](#-deployment-on-aws-ec2)
- [Alternative Deployment Options](#-alternative-deployment-options)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Disclaimer](#-disclaimer)

---

## 📖 Overview

This project tackles a critical healthcare problem: identifying high-risk cancer patients early. Given the sensitive nature of medical predictions, special focus was placed on **recall for the High-risk class** — missing a high-risk patient is far worse than a false alarm.

**Problem Type:** Multi-class Classification  
**Target Variable:** `Risk_Level` (Low / Medium / High)  
**Final Model:** Optuna-tuned, class-weighted XGBoost  
**Accuracy:** 88%  
**Macro F1-Score:** 0.72

---

## 📊 Dataset

The dataset contains patient records with 17 features:

**Demographics:** Age, Gender  
**Lifestyle:** Smoking, Alcohol_Use, Obesity, Physical_Activity, Physical_Activity_Level  
**Diet:** Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Calcium_Intake  
**Medical History:** Family_History, BRCA_Mutation, H_Pylori_Infection, BMI  
**Environmental:** Air_Pollution, Occupational_Hazards

**Class Distribution (imbalanced):**
- Medium: 1574 patients (~79%)
- Low: 324 patients (~16%)
- High: 102 patients (~5%)

The strong class imbalance motivated the use of **SMOTE** and **class weighting** during training.

---

## 🔄 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of risk factors across risk levels
- Correlation analysis to identify key predictors
- Visualization with countplots, KDE plots, and correlation bar charts

### 2. Data Preprocessing
- Encoded the `Risk_Level` target using `LabelEncoder`
- Applied one-hot encoding for categorical features
- **Removed leaky features:** `Cancer_Type` and `Overall_Risk_Score` (these directly encode the outcome and would inflate model performance unrealistically)

### 3. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set only
- Tested class weighting as an alternative in XGBoost
- Compared performance with and without resampling

### 4. Model Development
Tested multiple models with proper train/test split (stratified, 80/20):
- Logistic Regression
- Random Forest Classifier
- Random Forest + SMOTE
- Optuna-tuned Random Forest
- XGBoost (baseline)
- **Optuna-tuned class-weighted XGBoost ✅ (Winner)**

### 5. Hyperparameter Tuning
Used **Optuna** with the TPE sampler to search the hyperparameter space, optimizing macro-F1 score and High-class recall through cross-validation.

### 6. Deployment
Built an interactive **Streamlit web app** with manual input and batch CSV prediction modes, including visualizations (gauge charts, probability bars, pie charts).

---

## 💡 Key Findings

Through EDA and feature correlation analysis, the strongest indicators of High risk were:

- **Heavy smoking** is the strongest single predictor of High risk
- **High alcohol use** strongly correlates with High risk
- **High red meat consumption** is associated with elevated risk
- **High intake of salted/processed foods** strongly linked to High risk
- **Exposure to high air pollution** is a major differentiator
- **Hazardous occupational exposures** push risk higher
- **Higher obesity levels** correlate with higher risk

---

## 🎯 Model Performance

Final model: **Optuna-tuned class-weighted XGBoost**

| Metric    | High | Low  | Medium | Macro Avg |
|-----------|------|------|--------|-----------|
| Precision | 0.50 | 0.76 | 0.92   | 0.73      |
| Recall    | 0.45 | 0.78 | 0.92   | 0.72      |
| F1-Score  | 0.47 | 0.77 | 0.92   | 0.72      |

**Overall Accuracy:** 88%

The model successfully balances overall accuracy with strong recall on the clinically important High-risk class.

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn
- **Hyperparameter Tuning:** Optuna
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web App:** Streamlit
- **Model Persistence:** joblib
- **Deployment:** AWS EC2, Docker

---

## 📁 Project Structure

```
Cancer_Risk_Prediction/
├── app.py                          # Streamlit web application
├── cancer_risk_predictor.ipynb     # Full ML pipeline notebook
├── model_xgb_new.pkl               # Trained XGBoost model
├── label_encoder.pkl               # Fitted LabelEncoder
├── feature_names.pkl               # 17 feature names
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container deployment config
├── .gitignore                      # Files to exclude from Git
└── README.md                       # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

1. **Clone the repository**
```bash
   git clone https://github.com/Aryan09092001/Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML.git
   cd Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

---

## 🚀 Running the App Locally

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### App Features
- **Manual Input Mode:** Enter individual patient features through an interactive sidebar form
- **Batch Mode:** Upload a CSV of multiple patients for bulk predictions
- **Visual Outputs:** Probability bar charts, High-risk gauge, distribution pie charts
- **Clinical Interpretation:** Color-coded results with appropriate recommendations
- **Downloadable Results:** Export batch predictions to CSV

---

## ☁️ Deployment on AWS EC2

Follow these step-by-step commands to deploy the app on an AWS EC2 instance.

### Step 1: Launch an EC2 Instance

1. Sign in to the [AWS Management Console](https://aws.amazon.com/console/)
2. Navigate to **EC2 → Instances → Launch Instance**
3. Configure the instance:
   - **Name:** `cancer-risk-predictor`
   - **AMI:** Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance Type:** `t2.micro` (Free tier)
   - **Key Pair:** Create a new key pair, download the `.pem` file, and keep it safe
   - **Network Settings:** Allow **SSH (port 22)** and add a custom TCP rule for **port 8501** (Streamlit's default port) from anywhere (`0.0.0.0/0`)
4. Click **Launch Instance**

### Step 2: Connect to the EC2 Instance via SSH

Open your terminal (PowerShell on Windows or Terminal on Mac/Linux) and run:

```bash
# Set permissions on the key file (Mac/Linux only)
chmod 400 your-key.pem

# Connect via SSH (replace with your EC2 public IP)
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

> **Windows users:** If `chmod` doesn't work, use the PuTTY tool or simply use the command without `chmod`.

### Step 3: Update the System and Install Python

Once connected to the EC2 instance, run:

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install python3-pip python3-venv git -y

# Verify installation
python3 --version
pip3 --version
```

### Step 4: Clone Your Repository

```bash
# Clone the project from GitHub
git clone https://github.com/Aryan09092001/Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML.git

# Navigate into the project folder
cd Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML
```

### Step 5: Set Up Python Environment and Install Dependencies

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### Step 6: Run the Streamlit App

```bash
# Run with public network access
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Step 7: Access the App

Open your browser and visit:

```
http://<your-ec2-public-ip>:8501
```

🎉 Your app is now live!

### Step 8: Keep the App Running 24/7 (Optional)

The app stops when you close the SSH terminal. To keep it running permanently:

**Option A: Using `nohup` (simplest)**

```bash
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 > streamlit.log 2>&1 &
```

You can now safely close the SSH connection — the app will keep running.

**Option B: Using `tmux` (recommended)**

```bash
# Install tmux
sudo apt install tmux -y

# Start a new tmux session
tmux new -s streamlit

# Run the app inside the session
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Detach from the session (app keeps running)
# Press: Ctrl + B, then D

# To reattach later
tmux attach -t streamlit
```

**Option C: Using `systemd` (production-grade)**

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/streamlit.service
```

Paste the following (update paths to your username):

```ini
[Unit]
Description=Streamlit Cancer Risk Predictor
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML
ExecStart=/home/ubuntu/Predictive_Modeling_for_Cancer_Risk_Assessment_Using_ML/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Check status
sudo systemctl status streamlit
```

The app now restarts automatically if it crashes or the server reboots.

### Step 9: Stopping the App

```bash
# If using nohup
pkill -f streamlit

# If using tmux
tmux kill-session -t streamlit

# If using systemd
sudo systemctl stop streamlit
```

---

## 🐳 Alternative Deployment Options

### Option 1: Deploy with Docker

```bash
# Build the Docker image
docker build -t cancer-risk-predictor .

# Run the container
docker run -p 8501:8501 cancer-risk-predictor
```

Access at `http://localhost:8501`.

### Option 2: Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** and select this repo
5. Set the main file path to `app.py`
6. Click **"Deploy"** — done in 1 minute!

### Option 3: AWS Elastic Beanstalk

For managed AWS deployment without manual server setup:

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.11 cancer-risk-predictor

# Create environment and deploy
eb create cancer-risk-env

# Open the deployed app
eb open
```

---

## 📸 Screenshots

> _Replace these placeholders with your actual screenshots:_
> 
> ![Main Interface](images/main_interface.png)
> 
> ![Prediction Results](images/prediction_results.png)
> 
> ![Batch Prediction](images/batch_prediction.png)

To add screenshots:
1. Create a folder called `images` in your repo
2. Upload your screenshots there
3. Reference them in this README using `![Description](images/filename.png)`

---

## 🔮 Future Improvements

- [ ] Add **SHAP values** to explain individual predictions
- [ ] Include a **feature importance chart** in the app
- [ ] Implement **user authentication** for clinical use
- [ ] Add a **database backend** (PostgreSQL/MongoDB) to log predictions over time
- [ ] Improve **High-class recall** with cost-sensitive learning and threshold tuning
- [ ] Deploy with **HTTPS** using Nginx + Let's Encrypt
- [ ] Add a **custom domain** name
- [ ] Build a **REST API** version using FastAPI for system integration
- [ ] Add **A/B testing** between different model versions
- [ ] Set up **CI/CD pipeline** with GitHub Actions for automated deployment

---

## 👨‍💻 Author

**Aryan**  
GitHub: [@Aryan09092001](https://github.com/Aryan09092001)

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. The predictions made by this model are based on a limited dataset and should not be relied upon for clinical decision-making.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this project helpful, please consider giving it a star on GitHub!**
