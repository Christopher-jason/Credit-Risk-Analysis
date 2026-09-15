import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import loan_train, test_loan

# One-hot encode - shared across all three models
X = loan_train.drop(columns=['id', 'loan_status'])
y = loan_train['loan_status']
X = pd.get_dummies(X, columns=[
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
], drop_first=True)

test_loan_encoded = pd.get_dummies(test_loan, columns=[
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
], drop_first=True)
test_loan_encoded = test_loan_encoded.reindex(columns=X.columns, fill_value=0)


def print_results(model_name, y_val, y_val_pred):
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0         {cm[0][0]:<13} {cm[0][1]}")
    print(f"Actual 1         {cm[1][0]:<13} {cm[1][1]}")
    print("\nFull classification report:")
    print(classification_report(y_val, y_val_pred))
    return acc, f1


def run_logistic_regression(X, y, X_test=None, show_chart=True):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_val_pred = model.predict(X_val_scaled)
    print_results("Logistic Regression", y_val, y_val_pred)

    if show_chart:
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Logistic Regression - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    test_predictions = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        print(f"\nPredicted default rate on test_loan: {test_predictions.mean() * 100:.2f}%")

    return model, test_predictions


def run_random_forest(X, y, X_test=None, show_chart=True):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(class_weight='balanced', n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print_results("Random Forest", y_val, y_val_pred)

    if show_chart:
        cm = confusion_matrix(y_val, y_val_pred)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0])
        axes[0].set_title("Random Forest - Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        importances.plot(kind='barh', ax=axes[1])
        axes[1].set_title("Top 10 Feature Importances")
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.show()

    test_predictions = None
    if X_test is not None:
        test_predictions = model.predict(X_test)
        print(f"\nPredicted default rate on test_loan: {test_predictions.mean() * 100:.2f}%")

    return model, test_predictions


def run_xgboost(X, y, X_test=None, show_chart=True):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print_results("XGBoost", y_val, y_val_pred)

    if show_chart:
        cm = confusion_matrix(y_val, y_val_pred)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
        axes[0].set_title("XGBoost - Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        importances.plot(kind='barh', ax=axes[1])
        axes[1].set_title("Top 10 Feature Importances")
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.show()

    test_predictions = None
    if X_test is not None:
        test_predictions = model.predict(X_test)
        print(f"\nPredicted default rate on test_loan: {test_predictions.mean() * 100:.2f}%")

    return model, test_predictions


if __name__ == '__main__':
    lr_model, lr_test_preds = run_logistic_regression(X, y, X_test=test_loan_encoded)
    rf_model, rf_test_preds = run_random_forest(X, y, X_test=test_loan_encoded)
    xgb_model, xgb_test_preds = run_xgboost(X, y, X_test=test_loan_encoded)