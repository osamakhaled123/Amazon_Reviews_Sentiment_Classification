import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src import tokenizers
from src.classification_models import NN_Deep, deep_training, deep_predict

training_data = pd.read_csv('/data/cleaned_training_reviews.csv')
validating_data = pd.read_csv('/data/cleaned_validating_reviews.csv')

X_train, X_val = tokenizers.TF_IDF(training_data['cleaned'], validating_data['cleaned'], training_data['score'])

name = 'Deep'
model = NN_Deep(drop_out = 0.2)
filename = f"{name.replace(' ', '_').lower()}_model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('models/'+filename+'.pt'):
    epochs = 4
    batch_size = 64
    lr = 0.001

    model, train_losses, val_losses = deep_training(
        model=model,
        train_data=X_train,
        train_target=training_data['score']-1,
        val_data=X_val,
        val_target=validating_data['score']-1,
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    torch.save({'model_state_dict':model.state_dict(),
                'train_losses':train_losses,
                'val_losses':val_losses,
                'batch_size':batch_size,
                'learning_rate':lr
                }, 'models/'+filename+'.pt')

else:
    print("loading model...")
    checkpoint = torch.load('models/'+filename+'.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

y_train_pred = deep_predict(model=model, data=X_train, target=training_data['score'], batch_size=32)
y_val_pred = deep_predict(model=model, data=X_val, target=validating_data['score'], batch_size=32)


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