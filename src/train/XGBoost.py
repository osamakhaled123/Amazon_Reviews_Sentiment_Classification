import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src import tokenizers

training_data = pd.read_csv('/data/cleaned_training_reviews.csv')
validating_data = pd.read_csv('/data/cleaned_validating_reviews.csv')

name = 'XGBoost'
model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
filename = f"{name.replace(' ', '_').lower()}_model"

if not os.path.exists('models/'+filename+'.json'):
    X_train, X_val = tokenizers.TF_IDF(training_data['cleaned'], validating_data['cleaned'], training_data['score'])

    classes = np.unique(training_data['score'])
    weights = compute_class_weight('balanced', classes=classes, y=training_data['score'])
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = training_data['score'].map(class_weight_dict).values

    model.fit(X_train, training_data['score']-1, sample_weight=sample_weights)
    model.save_model('models/'+filename+'.json')

else:
    print("loading model...")
    model.load_model('models/xgboost_model.json')

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print(f"\n================== {name} ==================")
print(f"Training Accuracy:\t{accuracy_score(training_data['score'], y_train_pred):.4f}")
print(f"Validation Accuracy:\t{accuracy_score(validating_data['score'], y_val_pred):.4f}")
print("========================================================================================\n")
print(f"Training F1-score:\t{f1_score(training_data['score'], y_train_pred, average='weighted'):.4f}")
print(f"Validation F1-score:\t{f1_score(validating_data['score'], y_val_pred, average='weighted'):.4f}")
print("========================================================================================\n")
print("\nClassification Reports:")
print("Training:\n", classification_report(training_data['score'], y_train_pred))
print("Validation:\n", classification_report(validating_data['score'], y_val_pred))

# ---------- Confusion Matrix ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training confusion matrix
cm_train = confusion_matrix(training_data['score'], y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title(f"{name} - Training Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Validation confusion matrix
cm_val = confusion_matrix(validating_data['score'], y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap="Blues", ax=axes[1])
axes[1].set_title(f"{name} - Validation Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()