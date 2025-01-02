import streamlit as st
import pandas as pd
import joblib  # Για φόρτωση του μοντέλου

# Εμφάνιση τίτλου
st.title("Εφαρμογή Πρόβλεψης για Καρδιοαγγειακά Νοσήματα")

# Ορισμός των χαρακτηριστικών που χρησιμοποιούνται στην εκπαίδευση του μοντέλου
training_features = [
    'age', 'trestbps', 'chol', 'thalch',
    'cp_typical angina', 'cp_non-anginal', 'cp_atypical angina',
    'exang_True',
    'slope_flat', 'slope_upsloping'
]

# Ορισμός των κλιμάκων για κάθε χαρακτηριστικό
feature_ranges = {
    'age': (18, 100),
    'trestbps': (80, 200),
    'chol': (100, 600),
    'thalch': (60, 250),
    'cp_typical angina': (0.0, 1.0),  # 0 = Όχι, 1 = Ναι
    'cp_non-anginal': (0.0, 1.0),
    'cp_atypical angina': (0.0, 1.0),
    'exang_True': (0.0, 1.0),  # 0 = Όχι, 1 = Ναι (Δείχνει αν υπάρχει πόνος στο στήθος μετά την άσκηση)
    'slope_flat': (0.0, 1.0),  # 0 = Δεν είναι επίπεδη, 1 = Επίπεδη κλίση του ST segment (καρδιογράφημα)
    'slope_upsloping': (0.0, 1.0)  # 0 = Δεν είναι ανοδική, 1 = Ανοδική κλίση του ST segment (καρδιογράφημα)
}

# Περιγραφές για κάθε χαρακτηριστικό
feature_descriptions = {
    'age': "Η ηλικία του ατόμου σε χρόνια.",
    'trestbps': "Η πίεση του αίματος σε ηρεμία (σε mm Hg).",
    'chol': "Η χοληστερόλη (σε mg/dl).",
    'thalch': "Η μέγιστη καρδιακή συχνότητα κατά τη διάρκεια άσκησης (σε bpm).",
    'cp_typical angina': "Αν το άτομο έχει τυπικό πόνο θώρακα (1 = Ναι, 0 = Όχι).",
    'cp_non-anginal': "Αν το άτομο έχει μη ανγειακό πόνο θώρακα (1 = Ναι, 0 = Όχι).",
    'cp_atypical angina': "Αν το άτομο έχει μη τυπικό πόνο θώρακα (1 = Ναι, 0 = Όχι).",
    'exang_True': "Αν το άτομο έχει πόνο στο στήθος μετά από άσκηση (1 = Ναι, 0 = Όχι).",
    'slope_flat': "Αν η κλίση του ST segment στο καρδιογράφημα είναι επίπεδη (1 = Ναι, 0 = Όχι).",
    'slope_upsloping': "Αν η κλίση του ST segment στο καρδιογράφημα είναι ανοδική (1 = Ναι, 0 = Όχι)."
}

# Δημιουργία της φόρμας εισαγωγής μόνο για τα δεδομένα που χρησιμοποιούνται στην εκπαίδευση
input_data = {}

for feature in training_features:
    min_val, max_val = feature_ranges[feature]
    
    # Εμφάνιση της προτεινόμενης κλίμακας (στα Ελληνικά)
    st.write(f"Προτεινόμενη κλίμακα για το {feature}: {min_val} - {max_val}")
    st.write(f"Επεξήγηση: {feature_descriptions[feature]}")
    
    # Ορισμός του βήματος: Αν το χαρακτηριστικό είναι ακέραιο, το βήμα θα είναι 1, αλλιώς 0.1
    if isinstance(min_val, int):
        step_value = 1
    else:
        step_value = 0.1
    
    # Εισαγωγή τιμής από τον χρήστη
    value = st.number_input(f"Εισάγετε τιμή για το {feature}:",
                           min_value=min_val, max_value=max_val, step=step_value)

    # Αποθήκευση της τιμής στο λεξικό input_data
    input_data[feature] = value

# Φόρτωση του μοντέλου εκτός του Streamlit loop (έναρξη στο ξεκίνημα της εφαρμογής)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        st.error(f"Παρουσιάστηκε σφάλμα κατά τη φόρτωση του μοντέλου: {e}")
        return None

model = load_model()

# Συνάρτηση για εκτέλεση πρόβλεψης
def make_prediction(model, input_data):
    if model:
        try:
            # Μετατροπή των εισαγόμενων δεδομένων σε DataFrame
            input_df = pd.DataFrame([input_data])

            # Πρόβλεψη χρησιμοποιώντας το μοντέλο
            prediction = model.predict(input_df)

            # Εμφάνιση του αποτελέσματος
            if prediction[0] == 1:
                st.warning("Προειδοποίηση: Υψηλός κίνδυνος για καρδιοαγγειακά νοσήματα!")
            else:
                st.success("Ο κίνδυνος για καρδιοαγγειακά νοσήματα είναι χαμηλός.")

            # Υπολογισμός του ποσοστού
            proba = model.predict_proba(input_df)[:, 1]  # Προβλεπόμενη πιθανότητα για την κατηγορία '1'
            st.write(f"Πιθανότητα καρδιοαγγειακού κινδύνου: {proba[0]*100:.2f}%")

        except Exception as e:
            st.error(f"Παρουσιάστηκε σφάλμα κατά την πρόβλεψη: {e}")

# Δημιουργία του κουμπιού "Πρόβλεψη"
if st.button('Πρόβλεψη'):
    make_prediction(model, input_data)
