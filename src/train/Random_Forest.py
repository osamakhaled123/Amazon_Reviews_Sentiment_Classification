import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src import tokenizers


training_data = pd.read_csv('/data/cleaned_training_reviews.csv')
validating_data = pd.read_csv('/data/cleaned_validating_reviews.csv')

name = 'Random Forest'
filename = f"{name.replace(' ', '_').lower()}_model"

X_train, X_val = tokenizers.TF_IDF(training_data['cleaned'], validating_data['cleaned'], training_data['score'])

if not os.path.exists('models/'+filename+'.pkl'):
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200,
                                   max_depth=20, n_jobs=-1, max_features='sqrt', min_samples_leaf=5)

    model.fit(X_train, training_data['score'])
    joblib.dump(model, 'models/'+filename+'.pkl', compress=('gzip', 3))

else:
    print("loading model...")
    model = joblib.load('models/'+filename+'.pkl')


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