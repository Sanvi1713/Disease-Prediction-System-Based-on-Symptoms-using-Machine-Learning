# ===============================================================
# DISEASE PREDICTION SYSTEM (GUI + ML MODELS)
# ===============================================================

import warnings
warnings.filterwarnings("ignore")  # ✅ Permanently hide all UserWarnings

import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# ===============================================================
# STEP 1 — LOAD TRAINING DATA
# ===============================================================
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encode target variable
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Create mapping for decoded output
disease_mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

# Split data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# STEP 2 — TRAIN MODELS
# ===============================================================
rf_model = RandomForestClassifier(random_state=42)
nb_model = GaussianNB()
svm_model = SVC()

rf_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# ===============================================================
# STEP 3 — CALCULATE ACCURACY
# ===============================================================
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

accuracy_text = (
    f"Random Forest Accuracy: {rf_acc*100:.2f}%\n"
    f"Naive Bayes Accuracy: {nb_acc*100:.2f}%\n"
    f"SVM Accuracy: {svm_acc*100:.2f}%"
)
print(accuracy_text)

# ===============================================================
# STEP 4 — DISEASE PREDICTION FUNCTION
# ===============================================================
symptoms = X.columns.values
symptom_index = {}

for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

def predictDisease(symptoms):
    input_data = [0] * len(X.columns)
    for symptom in symptoms.split(","):
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in X.columns:
            input_data[X.columns.get_loc(symptom)] = 1

    # ✅ Wrap input in DataFrame (fix warnings)
    input_df = pd.DataFrame([input_data], columns=X.columns)

    rf_prediction = rf_model.predict(input_df)[0]
    nb_prediction = nb_model.predict(input_df)[0]
    svm_prediction = svm_model.predict(input_df)[0]

    votes = [rf_prediction, nb_prediction, svm_prediction]
    if votes.count(rf_prediction) >= 2:
        final_prediction = rf_prediction
    elif votes.count(nb_prediction) >= 2:
        final_prediction = nb_prediction
    else:
        final_prediction = svm_prediction

    return {
        "rf_model_prediction": disease_mapping[int(rf_prediction)],
        "naive_bayes_prediction": disease_mapping[int(nb_prediction)],
        "svm_model_prediction": disease_mapping[int(svm_prediction)],
        "final_prediction": disease_mapping[int(final_prediction)]
    }

# ===============================================================
# STEP 5 — BUILD GUI (Tkinter)
# ===============================================================
root = tk.Tk()
root.title("🧠 Disease Prediction System")
root.geometry("600x550")
root.configure(bg="#F4F6F7")

# Title
title_label = tk.Label(
    root, text="DISEASE PREDICTION SYSTEM",
    font=("Arial", 16, "bold"), bg="#1ABC9C", fg="white", pady=10
)
title_label.pack(fill="x")

# Accuracy Display
accuracy_label = tk.Label(
    root, text=accuracy_text, font=("Arial", 11),
    fg="green", bg="#F4F6F7", justify="left"
)
accuracy_label.pack(pady=10)

# Input Label
symptom_label = tk.Label(
    root, text="Enter Symptoms (comma-separated):",
    font=("Arial", 12), bg="#F4F6F7"
)
symptom_label.pack(pady=10)

# Input Field
symptom_entry = tk.Entry(root, width=60, font=("Arial", 12))
symptom_entry.pack(pady=5)

# Output Box
result_box = tk.Text(root, height=8, width=70, font=("Arial", 11))
result_box.pack(pady=10)

# Predict Button
def on_predict():
    symptoms = symptom_entry.get()
    if not symptoms:
        messagebox.showwarning("Input Error", "Please enter symptoms!")
        return
    result = predictDisease(symptoms)
    result_text = (
        f"\n🌿 Random Forest: {result['rf_model_prediction']}\n"
        f"🧬 Naive Bayes: {result['naive_bayes_prediction']}\n"
        f"⚙️ SVM: {result['svm_model_prediction']}\n\n"
        f"✅ Final Predicted Disease: {result['final_prediction']}"
    )
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, result_text)

predict_button = tk.Button(
    root, text="🔍 Predict Disease", command=on_predict,
    bg="#1ABC9C", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5
)
predict_button.pack(pady=10)

# Run GUI
root.mainloop()
