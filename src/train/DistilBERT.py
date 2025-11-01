import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification
from src.tokenizers import D_BERT_pre_processing
from src.classification_models import DistilBert_train, DistilBert_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
from matplotlib import pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')

training_data = pd.read_csv('/data/cleaned_training_reviews.csv')
validating_data = pd.read_csv('/data/cleaned_validating_reviews.csv')

training_data.rename(columns={'score':'labels'}, inplace=True)
validating_data.rename(columns={'score':'labels'}, inplace=True)

name = 'DistilBERT'
filename = f"{name.replace(' ', '_').lower()}_model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('models/'+filename+'.pt'):
    epochs = 1
    batch_size = 64
    lr = 0.005

    num_classes = training_data["labels"].nunique()

    train_dataloader, val_dataloader = D_BERT_pre_processing(training_data, validating_data, batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(
        "/content/drive/MyDrive/distilbert_local")

    model, train_losses, val_losses = DistilBert_train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=epochs,
        learning_rate=lr,
    )

    torch.save({'model_state_dict':model.state_dict(),
                'train_losses':train_losses,
                'val_losses':val_losses,
                'batch_size':batch_size,
                'learning_rate':lr,
                'num_classes':num_classes
                }, 'models/'+filename+'.pt')


else:
    checkpoint = torch.load('models/'+filename+'.pt')
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=checkpoint['num_classes'])
    print("loading model...")
    model.load_state_dict(checkpoint['model_state_dict'])

y_train_pred, y_train_gt = DistilBert_predict(model=model, val_dataloader=train_dataloader, device=device),
y_val_pred, y_val_gt = DistilBert_predict(model=model, val_dataloader=val_dataloader, device=device)


print(f"\n================== {name} ==================")
print(f"Training Accuracy:\t{accuracy_score(y_train_gt, y_train_pred):.4f}")
print(f"Validation Accuracy:\t{accuracy_score(y_val_gt, y_val_pred):.4f}")
print("========================================================================================\n")
print(f"Training F1-score:\t{f1_score(y_train_gt, y_train_pred, average='weighted'):.4f}")
print(f"Validation F1-score:\t{f1_score(y_val_gt, y_val_pred, average='weighted'):.4f}")
print("========================================================================================\n")
print("\nClassification Reports:")
print("Training:\n", classification_report(y_train_gt, y_train_pred))
print("Validation:\n", classification_report(y_val_gt, y_val_pred))

# ---------- Confusion Matrix ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training confusion matrix
cm_train = confusion_matrix(y_train_gt, y_train_pred)
sns.heatmap(cm_train, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title(f"{name} - Training Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Validation confusion matrix
cm_val = confusion_matrix(y_val_gt, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap="Blues", ax=axes[1])
axes[1].set_title(f"{name} - Validation Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()