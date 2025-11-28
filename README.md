# 🧠 Stress Level Estimator

A machine-learning powered web application that predicts **daily stress level (0–10)** from physical activity metrics such as steps, active minutes, and calories burned. Built using **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 📘 Overview
This project trains a Linear Regression model on Fitbit’s daily activity dataset and uses a **synthetic stress scoring formula** to infer stress from activity levels.  
A clean and interactive Streamlit UI allows users to simulate daily activity and instantly get their predicted stress level.

The app also logs each prediction to `stress_log.csv` for future analysis.

---

## 🚀 Features

### ✔ Machine Learning Model
- Linear Regression trained using activity metrics  
- Synthetic stress calculation ensures realistic predictions  
- Model saved as `stress_model.pkl`

### ✔ Beautiful Streamlit Interface
- Sliders for steps & activity minutes  
- Auto-calculated calories  
- Stress gauge bar visualization  
- Sidebar with:
  - 💡 Tip of the Day  
  - 📈 Average reference activity values  

### ✔ Logging
- Automatically logs inputs and predictions to `stress_log.csv`

---

## 🧩 Architecture Diagram
```
                +-------------------------+
                |   dailyActivity CSV     |
                +-----------+-------------+
                            |
                            v
                 +----------+----------+
                 |  Preprocessing &    |
                 | Synthetic Stress    |
                 | Level Generation    |
                 +----------+----------+
                            |
                            v
                +-------------------------+
                |   Linear Regression     |
                |     Model Training      |
                +-----------+-------------+
                            |
                (stress_model.pkl saved)
                            |
                            v
       +---------------------------------------------+
       |                 Streamlit App                |
       +-------------------+-------------------------+
                           |
       +---------------------------------------------+
       | User Inputs → Model Prediction → UI Output  |
       +---------------------------------------------+
```

---

## 📦 Installation & Setup

### **1️⃣ Clone this Repository**
```bash
git clone https://github.com/yourusername/stress-estimator.git
cd stress-estimator
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Train the Model (optional)**
```bash
python train_model.py
```
This generates:
```
stress_model.pkl
```

### **4️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 📊 Synthetic Stress Formula

```
Active Score  = (VeryActive × 2)
              + (FairlyActive × 1.5)
              + (LightlyActive × 1)

Calorie Score = Calories / 10
Step Score    = Steps / 100

Activity Score = Active Score + Calorie Score + Step Score

Stress Level = 10 − (Activity Score / 100)
```

🔹 High activity → **lower stress**  
🔹 Low activity → **higher stress**  
🔹 Output range: **0 to 10**

---

## 🖥 User Interface Overview

### Main Screen
- Sliders for steps & minutes  
- Auto calorie estimation  
- Stress prediction  
- Stress gauge chart  
- Personalized message based on stress

### Sidebar
- **💡 Tip of the Day**  
- **📈 Average Values for Reference**

---

## 📁 Project Structure
```
│── stress_model.pkl
│── app.py
│── train_model.py
│── dailyActivity_merged_cleaned.csv
│── stress_log.csv
│── README.md
│── requirements.txt
```

---

## 🧪 Sample Workflow
1. User sets:
   - Steps: 8000  
   - Very Active: 30 min  
   - Fairly Active: 20 min  
   - Lightly Active: 60 min  
2. Calories auto-calculated  
3. Model predicts something like:
   ```
   Stress Level = 3.5 / 10
   ```
4. App displays a message and bar gauge.

---

## 📈 Future Improvements
- Add heart-rate, sleep & HRV data  
- Deploy on Streamlit Cloud  
- Replace Linear Regression with Random Forest / XGBoost  
- Add history dashboards and charts  

---

## 🤝 Contributing
Contributions are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📜 License
MIT License © 2025  
Made with ❤️ by Bhoomika.

