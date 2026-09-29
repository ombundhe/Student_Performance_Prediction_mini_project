# Student Performance Prediction — Professional Dashboard

## Project Goal
Predict whether a student will Pass or Fail using:
- Study Hours
- Attendance
- Internal Marks
- Assignment Marks
- Previous Performance

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Random Forest, Streamlit, Plotly, Joblib.

## Run on Mac / Windows / Linux

### 1. Open Terminal / Command Prompt
Go into the project folder.

### 2. Create a virtual environment
Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Start dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Dashboard Pages
1. Overview
2. Prediction
3. Student Analytics
4. Dataset
5. Model Performance
6. Insights
7. About Project

## Note
The included CSV is synthetic demonstration data for an academic project. Replace it with approved institutional data if available, and retrain the model.
