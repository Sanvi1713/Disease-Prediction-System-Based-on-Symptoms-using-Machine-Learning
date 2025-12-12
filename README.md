Disease Prediction System Based on Symptoms Using Machine Learning
A machine-learning powered system for early disease detection based on symptoms.

Overview
This project is a Disease Prediction System that uses Machine Learning (ML) algorithms to predict possible diseases based on user-entered symptoms. The system processes a custom dataset, trains multiple ML models, and provides predictions through a user-friendly Python GUI application.
The goal is to help support early disease identification and assist healthcare processes through fast, data-driven predictions.

Features
✔ Predicts diseases based on user-input symptoms
✔ Cleaned and preprocessed dataset for training
✔ Multiple ML models trained (Random Forest, Naive Bayes, SVM, etc.)
✔ High accuracy results with cross-validation
✔ GUI built using Python (Tkinter)
✔ Training and test CSV files included
✔ Easy to run and extend

Machine Learning Workflow

1. Data Preprocessing

   * Handling missing values
   * Label encoding
   * Feature selection
   * Data normalization

2. Model Training

   * Random Forest Classifier
   * Naive Bayes
   * Support Vector Classifier
   * Gradient Boosting
   * Decision Tree

3. Model Evaluation

   * Accuracy Score
   * Confusion Matrix
   * Cross-validation

4. Prediction

   * GUI-based symptom input
   * Model-based prediction output

Project Structure
Disease-Prediction-System-Based-on-Symptoms-using-Machine-Learning/
│
├── Training.csv
├── Testing.csv
├── create_sample_data.py
├── disease_prediction_gui.py
├── README.md
└── requirements.txt   (optional if you want me to generate)


How to Run the Project

1. Clone the repository

git clone https://github.com/Sanvi1713/Disease-Prediction-System-Based-on-Symptoms-using-Machine-Learning.git


2. Install dependencies

pip install -r requirements.txt



3. Run the GUI

python disease_prediction_gui.py


Technologies Used

* Python
* Machine Learning (scikit-learn)
* NumPy, Pandas
* Matplotlib / Seaborn
* Tkinter (GUI)

Dataset

* Training.csv: Contains symptoms and corresponding disease labels
* Testing.csv: Used to evaluate model accuracy
* Custom dataset designed for multi-class disease classification.

Future Enhancements

* Add deep learning support
* Deploy as a web application
* Add more diseases
* Add medical history and patient data
* Integrate with APIs for real-time diagnosis

Contributing

Contributions are welcome!
Feel free to submit issues or pull requests to enhance the project.

License

This project is open-source and available under the MIT License.


