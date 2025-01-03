import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib  # Για την αποθήκευση του μοντέλου
from tkinter import Tk, filedialog

# Λειτουργία για επιλογή αρχείου
def select_file(prompt):
    root = Tk()
    root.withdraw()  # Απόκρυψη του παραθύρου Tkinter
    file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV files", "*.csv")])
    return file_path

# Λειτουργία για επιλογή φακέλου
def select_folder(prompt):
    root = Tk()
    root.withdraw()  # Απόκρυψη του παραθύρου Tkinter
    folder_path = filedialog.askdirectory(title=prompt)
    return folder_path

# Ζητάμε από τον χρήστη να επιλέξει το αρχείο CSV με τα δεδομένα
file_path = select_file("Επιλέξτε το αρχείο CSV με τα δεδομένα εκπαίδευσης")
if not file_path:
    print("Δεν επιλέξατε αρχείο. Τερματισμός εφαρμογής.")
    exit()

# Φόρτωση δεδομένων
try:
    data = pd.read_csv(file_path)
    print("Δεδομένα φορτώθηκαν με επιτυχία!")
except Exception as e:
    print(f"Σφάλμα κατά τη φόρτωση των δεδομένων: {e}")
    exit()

# Εμφάνιση πρώτων γραμμών και βασικών πληροφοριών
print(data.head())
print(data.info())

# Αφαίρεση ακραίων τιμών
outliers_thresholds = {
    'age': data['age'].quantile(0.99),
    'chol': data['chol'].quantile(0.99),
    'trestbps': data['trestbps'].quantile(0.99),
}

for col, threshold in outliers_thresholds.items():
    data = data[data[col] <= threshold]

# Επιλογή μόνο των χαρακτηριστικών που θα χρησιμοποιηθούν για εκπαίδευση
selected_columns = ['age', 'trestbps', 'chol', 'thalch', 
                    'cp_typical angina', 'cp_non-anginal', 'cp_atypical angina', 
                    'exang_True', 'slope_flat', 'slope_upsloping']

# Διασφάλιση ότι όλες οι επιλεγμένες στήλες υπάρχουν στα δεδομένα
data = data[selected_columns + ['num']]  # Επιλέγουμε και τον στόχο

# Μετατροπή κατηγορικών δεδομένων
data = pd.get_dummies(data, drop_first=True)

# Αντικατάσταση NaN με τον μέσο όρο της κάθε στήλης
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Διαχωρισμός χαρακτηριστικών και στόχου
X = data_imputed.drop(columns=['num'])
y = data_imputed['num']

# Ισοστάθμιση δεδομένων με SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Διαχωρισμός εκπαίδευσης και δοκιμής
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Κανονικοποίηση δεδομένων
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Δημιουργία και εκπαίδευση μοντέλου
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# Ζητάμε από τον χρήστη να επιλέξει φάκελο αποθήκευσης του μοντέλου
folder_path = select_folder("Επιλέξτε το φάκελο όπου θα αποθηκευτεί το μοντέλο")
if not folder_path:
    print("Δεν επιλέξατε φάκελο. Τερματισμός εφαρμογής.")
    exit()

# Αποθήκευση του εκπαιδευμένου μοντέλου σε αρχείο 'model.pkl'
model_path = f"{folder_path}/model.pkl"
joblib.dump(model, model_path)
print(f"Το μοντέλο αποθηκεύτηκε με επιτυχία στο '{model_path}'.")

# Πρόβλεψη στο test set
y_pred = model.predict(X_test_scaled)

# Υπολογισμός ακριβείας
accuracy = accuracy_score(y_test, y_pred)
print(f"Ακρίβεια στο Test Set: {accuracy:.4f}")

# Υπολογισμός άλλων μετρικών
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Υπολογισμός AUC-ROC
if len(np.unique(y_test)) == 2:
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
else:
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class='ovr')

print(f"\nROC AUC: {roc_auc:.4f}")
