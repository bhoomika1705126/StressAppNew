import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r'dailyActivity_merged_cleaned.csv')

# Select required features
df = df[['TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']]

# Define synthetic stress level formula (activity → lower stress)
def generate_stress(row):
    active_score = row['VeryActiveMinutes'] * 2 + row['FairlyActiveMinutes'] * 1.5 + row['LightlyActiveMinutes'] * 1
    calorie_score = row['Calories'] / 10
    step_score = row['TotalSteps'] / 100
    activity_score = active_score + calorie_score + step_score
    stress = 10 - (activity_score / 100)
    return max(0, min(10, stress))

df['StressLevel'] = df.apply(generate_stress, axis=1)

# Split and train model
X = df[['TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']]
y = df['StressLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'stress_model.pkl')

print("Model trained and saved successfully!")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random

# Load the trained model
model = joblib.load('stress_model.pkl')

st.set_page_config(page_title="Stress Estimator", layout="centered")

# ─────────────────────
# 📊 Sidebar Info
# ─────────────────────

# 💡 Tip of the Day
tips = [
    "Take 5-minute deep breathing breaks.",
    "A short walk can reset your mind.",
    "Try a no-screen break every hour.",
    "Stretch every 2 hours to reduce tension.",
    "Drink enough water today!"
]
st.sidebar.title("💡 Tip of the Day")
st.sidebar.write(random.choice(tips))

# 📈 Average Comparison
avg_vals = {
    "steps": 7500,
    "calories": 2100,
    "very_active": 25,
    "fairly_active": 20,
    "lightly_active": 60
}
st.sidebar.title("📈 Averages for Reference")
st.sidebar.markdown(f"- **Steps**: {avg_vals['steps']}")
st.sidebar.markdown(f"- **Calories**: {avg_vals['calories']}")
st.sidebar.markdown(f"- **Very Active**: {avg_vals['very_active']} min")
st.sidebar.markdown(f"- **Fairly Active**: {avg_vals['fairly_active']} min")
st.sidebar.markdown(f"- **Lightly Active**: {avg_vals['lightly_active']} min")

# ─────────────────────
# 🔍 Stress Predictor UI
# ─────────────────────

st.title("🧠 Stress Level Estimator based on Daily Activity")
st.write("Move the sliders to simulate your daily activity.")

# User Inputs
steps = st.slider('🚶 Total Steps', 0, 30000, 8000)
very_active = st.slider('🏃 Very Active Minutes', 0, 180, 30)
fairly_active = st.slider('🚴 Fairly Active Minutes', 0, 180, 20)
lightly_active = st.slider('🚶‍♂️ Lightly Active Minutes', 0, 300, 60)

# 🔁 Dynamic Calories Calculation
calories = int((very_active * 10) + (fairly_active * 8) + (lightly_active * 4) + (steps * 0.03))
st.markdown(f"🔥 **Estimated Calories Burned**: `{calories}`")

# Predict Button
if st.button('🔍 Predict Stress Level'):
    input_data = np.array([[steps, calories, very_active, fairly_active, lightly_active]])
    stress_level = model.predict(input_data)[0]
    stress_level = round(stress_level, 2)

    # Display result
    st.success(f"Predicted Stress Level: {stress_level} / 10")

    # Feedback message
    if stress_level > 6:
        st.warning("⚠️ High Stress. Consider more physical activity or breaks.")
    elif stress_level < 3:
        st.info("✅ Low Stress. Keep up the healthy routine!")
    else:
        st.write("🧘 Moderate Stress. Balance your activity and rest.")

    # 📊 Stress Gauge Bar Chart
    st.subheader("📊 Stress Gauge")
    gauge = pd.DataFrame({'Stress': [stress_level], 'Remaining': [10 - stress_level]})
    st.bar_chart(gauge)

    # 📝 Logging to CSV
    log_data = {
        "Steps": steps,
        "Calories": calories,
        "VeryActive": very_active,
        "FairlyActive": fairly_active,
        "LightlyActive": lightly_active,
        "StressLevel": stress_level
    }
    log_df = pd.DataFrame([log_data])
    try:
        log_df.to_csv("stress_log.csv", mode='a', header=False, index=False)
    except:
        pass
