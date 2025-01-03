import streamlit as st
import pandas as pd
import joblib  # Για φόρτωση του μοντέλου

# Εμφάνιση τίτλου
st.title("Εφαρμογή Πρόβλεψης για Καρδιοαγγειακά Νοσήματα")

# Χαρακτηριστικά και ορισμοί
training_features = [
    'age', 'trestbps', 'chol', 'thalch',
    'cp_typical angina', 'cp_non-anginal', 'cp_atypical angina',
    'exang_True',
    'slope_flat', 'slope_upsloping'
]

feature_ranges = {
    'age': (18, 100),
    'trestbps': (80, 200),
    'chol': (100, 600),
    'thalch': (60, 250),
    'cp_typical angina': (0.0, 1.0),
    'cp_non-anginal': (0.0, 1.0),
    'cp_atypical angina': (0.0, 1.0),
    'exang_True': (0.0, 1.0),
    'slope_flat': (0.0, 1.0),
    'slope_upsloping': (0.0, 1.0)
}

# Περιγραφές χαρακτηριστικών
feature_descriptions = {
    'age': "Η ηλικία του ατόμου σε χρόνια.",
    'trestbps': "Η πίεση του αίματος σε ηρεμία (σε mm Hg).",
    'chol': "Η χοληστερόλη (σε mg/dl).",
    'thalch': "Η μέγιστη καρδιακή συχνότητα κατά τη διάρκεια άσκησης (σε bpm).",
    'cp_typical angina': "Αν το άτομο έχει τυπικό πόνο θώρακα.",
    'cp_non-anginal': "Αν το άτομο έχει μη ανγειακό πόνο θώρακα.",
    'cp_atypical angina': "Αν το άτομο έχει μη τυπικό πόνο θώρακα.",
    'exang_True': "Αν το άτομο έχει πόνο στο στήθος μετά από άσκηση.",
    'slope_flat': "Αν η κλίση του ST segment στο καρδιογράφημα είναι επίπεδη.",
    'slope_upsloping': "Αν η κλίση του ST segment στο καρδιογράφημα είναι ανοδική."
}

# Εισαγωγή δεδομένων
input_data = {}
for feature in training_features:
    st.markdown(f"### {feature}:")
    st.markdown(f"- **Προτεινόμενη κλίμακα**: {feature_ranges[feature]}")
    st.markdown(f"- **Επεξήγηση**: {feature_descriptions[feature]}")
    min_val, max_val = feature_ranges[feature]
    value = st.number_input(f"Δώστε τιμή για {feature}:",
                            min_value=min_val, max_value=max_val, step=0.1 if isinstance(min_val, float) else 1)
    input_data[feature] = value

# Φόρτωση μοντέλου
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        st.error(f"Σφάλμα στη φόρτωση του μοντέλου: {e}")
        return None

model = load_model()

# Πρόβλεψη
def make_prediction(model, input_data):
    if model:
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[:, 1][0]
            if prediction == 1:
                st.error(f"Υψηλός κίνδυνος! Πιθανότητα: {proba*100:.2f}%")
            else:
                st.success(f"Χαμηλός κίνδυνος. Πιθανότητα: {proba*100:.2f}%")
        except Exception as e:
            st.error(f"Σφάλμα στην πρόβλεψη: {e}")

if st.button('Πρόβλεψη'):
    make_prediction(model, input_data)
