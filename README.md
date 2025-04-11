# Crime Prediction and Explainability Model using Random Forest

This project presents an end-to-end machine learning solution for predicting binary outcomes (e.g., crime classification) using a dataset structured similarly to the **IFSSA (Integrated Fire and Safety Services Agency)** dataset. It includes data preprocessing, model training using Random Forest, explainability via SHAP and LIME, and deployment with Flask. The solution is built to demonstrate the integration of machine learning into real-world decision-making workflows.

---

## Dataset Overview

The dataset used mimics the structure and logic of the **IFSSA** dataset, which typically includes variables like:
- **REPORT_MONTH**: Categorical time variable
- **WEEKDAY**: Day of the week
- **NEIGHBOURHOOD**, **INCIDENT**, etc.

It was used to build a classification model aimed at predicting binary class outcomes, such as identifying critical vs non-critical incidents. The dataset was preprocessed and split into training and testing subsets for model development.

---

## Exploratory Data Analysis (EDA)

The EDA phase revealed:
- Temporal patterns in incidents (e.g., spikes in certain months or days)
- Class imbalance in target variables
- Feature correlations and distributions
- Identification of categorical vs. continuous features

Visualization libraries such as Seaborn and Matplotlib were used to explore trends and relationships within the data.

---

## Machine Learning Model

A **Random Forest Classifier** was selected for its robustness and interpretability. Model training included:
- Handling categorical features (`LabelEncoder`)
- Feature importance analysis
- Hyperparameter tuning (optional)

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

---

## Model Iteration Summary

Multiple iterations were carried out to:
- Tune model performance
- Handle class imbalance
- Improve generalization and minimize overfitting

Final model achieved strong performance metrics (example):
```
Accuracy: 92%
Precision: 91%
Recall: 94%
F1-score: 92%
```

---

## Model Explainability

Model predictions were explained using:

### 🔹 SHAP (SHapley Additive exPlanations)
- Global feature importance
- Local instance interpretation using `force_plot`

### 🔹 LIME (Local Interpretable Model-agnostic Explanations)
- Individual prediction explanation with visual interface

Both tools were used to validate model transparency and feature influence.

---

## Deployment Process

The trained model was deployed using a **Flask API**:

```bash
POST /predict
Content-Type: application/json
Body: JSON object with feature data
```

The API loads the `random_forest_model.pkl` file using `joblib` and returns predictions in JSON format.

---

## Model Demo

A working demo showcases:
- Sending a test sample to the endpoint
- Real-time model inference
- Return of prediction results via API

---

## Use Cases

- Predicting critical crime types in real-time
- Assisting emergency response prioritization
- Supporting public safety policy decisions

---

## Recommendations for Future Improvements

- Integrate advanced model tuning (GridSearchCV, cross-validation)
- Handle imbalanced classes using SMOTE or class weighting
- Deploy via Docker or Streamlit for interactive dashboard
- Expand model to multi-class or multi-label problems

---

## Conclusion

This project demonstrates the power of combining machine learning, explainability, and deployment to solve public safety problems similar to those addressed by IFSSA. Key takeaways include:
- The importance of EDA in understanding real-world datasets
- The need for interpretable models in safety-critical domains
- How to bridge model development with deployment and accessibility

---

## 📁 File Structure

```
├── model_training.ipynb         # Model training, SHAP & LIME analysis
├── app.py                       # Flask API for deployment
├── random_forest_model.pkl      # Trained model
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── images/                      # Visual assets (SHAP plots, LIME screenshots)
```

---

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

