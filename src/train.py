import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import torch
import joblib
import tqdm
import tokenizers
import classification
import os


training_data = pd.read_csv('/data/cleaned_training_reviews.csv')
validating_data = pd.read_csv('/data/cleaned_validating_reviews.csv')

training_data['cleaned'] = training_data['cleaned'].astype(str)
validating_data['cleaned'] = validating_data['cleaned'].astype(str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unique, counts = np.unique(training_data['score'].values, return_counts=True)
class_weights = 1 / counts

X_train, emb_train_matrix, w2v_trained_model, word_to_index = tokenizers.word2vec(training_data['cleaned'])
X_val, emb_val_matrix, _, _ = tokenizers.word2vec(validating_data['cleaned'], w2v_trained_model, word_to_index)

models = {'Random Forest':RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200,
                                                 max_depth=20, n_jobs=-1, max_features='sqrt', min_samples_leaf=5),
          'Logistic Regression':LogisticRegression(class_weight='balanced', multi_class='multinomial',
                                                   solver="saga", max_iter=1000),
          'SVC':SVC(kernel='rbf', class_weight='balanced', max_iter=1000, C=0.2, random_state=42, probability=True),
          'XGBoost':XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42),
          'Deep': NN_Deep(drop_out = 0.2),
          'GRU':GRUClassifier(emb_matrix, 10, 5)}


for name, model in tqdm.tqdm(models.items(), desc="Training models", unit="model"):
    filename = f"{name.replace(' ', '_').lower()}_model"

    if name == 'XGBoost':
        classes = np.unique(training_data['score'])
        weights = compute_class_weight('balanced', classes=classes, y=training_data['score'])

        class_weight_dict = dict(zip(classes, weights))

        sample_weights = training_data['score'].map(class_weight_dict).values

        if not os.path.exists('models/'+filename+'.json'):
            model.fit(X_train, training_data['score'], sample_weight=sample_weights)
            model.save_model('models/'+filename+'.json')
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

    elif name == 'Deep':

        if not os.path.exists('models/'+filename+'.pt'):
            epochs = 1
            batch_size = 32
            lr = 0.001

            model, train_losses, val_losses = deep_training(
                mode=model,
                train_data=X_train,
                train_target=training_data['score'],
                val_data=X_val,
                val_target=validating_data['score'],
                learning_rate=lr,
                num_epochs=epochs,
                batch_size=batch_size
            )

            torch.save({'model_state_dict':model.state_dict(),
                        'train_losses':train_losses,
                        'val_losses':val_losses,
                        'batch_size':batch_size,
                        'learning_rate':lr
                        }, 'models/'+filename+'.pt')

            y_train_pred = deep_predict(model=model, data=X_train, target=training_data['score'], batch_size=32)
            y_val_pred = deep_predict(model=model, data=X_val, target=validating_data['score'], batch_size=32)

    else:
        if not os.path.exists('models/'+filename+'.pkl'):
            model.fit(X_train, training_data['score'])
            joblib.dump(model, 'models/'+filename+'.pkl', compress=('gzip', 3))
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

    print("=======================================================================================================================\n")
