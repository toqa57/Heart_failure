# Heart_failure

💓 Heart Disease Prediction with SVM and Decision Tree (Bagging Ensemble)
This project is a machine learning pipeline for predicting heart disease based on patient health data. It involves preprocessing, visualization, and classification using Support Vector Machine (SVM) and Decision Tree models with and without Bagging ensemble techniques.

📁 Dataset
The dataset contains anonymized health-related attributes such as:

- Age

- Sex

- ChestPainType

- RestingBP

- Cholesterol

- FastingBS

- RestingECG

- MaxHR

- ExerciseAngina

- Oldpeak

- ST_Slope

- HeartDisease (Target)

Note: This project assumes the dataset is in CSV format.

⚙️ Features
Z-score normalization

Label encoding for categorical variables

Train-Test split

SVM and Decision Tree classifiers

Bagging ensemble with SVM and Decision Tree

Evaluation using accuracy score

Visualization using Seaborn

🧠 Models Used
SVC()

DecisionTreeClassifier()

BaggingClassifier(estimator=SVC())

BaggingClassifier(estimator=DecisionTreeClassifier())

📊 Visualizations
Heatmap of feature correlations

Bar chart comparing model accuracies
