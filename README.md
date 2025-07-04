# 🍷 Wine Quality Predictor using Random Forest Classifier

---

## 🧠 How It Works

1. The model is trained using a labeled dataset of wine qualities.
2. Missing inputs are imputted using KNNImpute
3. The RandomForestClassifier predicts if the wine is of good quality or not.
4. Users can input their own wine chemical properties in the app to see predictions in real time.

---

## 🛠 Features

- ✔️ Real-time Wine Quality classification
- ✔️ Clean and minimal interface with Streamlit
- ✔️ Uses **StandardScaler** for feature extraction
- ✔️ Lightweight and fast with **Random Forest Classifier**


---

## ⚙️ Run Locally

1. Clone the repository
    ```bash
    git clone https://github.com/Pawieee/wine_prediction.git
    cd wine_prediction
    ```

2. Install dependencies
    ```bash
    pip install -r "requirements.txt"
    ```

3. Run streamlit
    ```bash
    streamlit run app.py
    ```

## Requirements

    - Python installed locally
    - Pandas
    - Scikit-learn
    - Streamlit
    - Joblib