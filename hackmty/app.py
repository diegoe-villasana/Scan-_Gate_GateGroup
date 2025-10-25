# ============================================
# 🌐 API Flask para predicción de consumo
# ============================================

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ============================================
# ⚙️ Inicialización
# ============================================

app = Flask(__name__)

# Cargar modelo y preprocesador
MODEL_PATH = "model/prediccion_consumo_REAL_ADV_CORR_v1.keras"
PREPROC_PATH = "model/preprocesador_consumo_REAL_ADV_CORR_v1.joblib"

model = keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROC_PATH)

print("✅ Modelo y preprocesador cargados correctamente")

# ============================================
# 🏠 Ruta principal (interfaz simple)
# ============================================

@app.route('/')
def home():
    return render_template('index.html')

# ============================================
# 🔮 Endpoint de predicción
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Convertir datos a DataFrame
        df = pd.DataFrame([data])

        # Preprocesar igual que en entrenamiento
        X_p = preprocessor.transform(df)

        # Hacer predicción
        y_pred = model.predict(X_p)
        y_pred_real = np.expm1(y_pred).flatten()[0]  # revertir log-transform

        return jsonify({
            "prediction": round(float(y_pred_real), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def home():

    predicciones = [10, 15, 12, 18, 20, 14]

    fig, ax = plt.subplots()
    ax.plot(predicciones, marker='o')
    ax.set_title("Predicciones de Consumo")
    ax.set_xlabel("Producto")
    ax.set_ylabel("Cantidad Predicha")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()  # Convertir a base64
    plt.close(fig)

    return render_template("index.html", plot_url=plot_url)

# ============================================
# 🚀 Run local
# ============================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    print("corriendo en {port}")
