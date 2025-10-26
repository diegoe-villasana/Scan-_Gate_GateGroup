from flask import Flask, render_template
import pandas as pd
import joblib
from tensorflow import keras

app = Flask(__name__)

# Cargar modelo y preprocesador
MODEL_PATH = "model/prediccion_consumo_REAL_ADV_CORR_v1.keras"
PREPROC_PATH = "model/preprocesador_consumo_REAL_ADV_CORR_v1.joblib"

model = keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROC_PATH)

# Cargar datos originales
df_original = pd.read_excel("data\[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx")

@app.route("/")
def home():
    # Preparar features
    df_analisis = df_original.copy()
    df_analisis['Product_Category'] = df_analisis['Product_ID'].str.slice(0,3)
    
    features = ['Origin', 'Flight_Type', 'Service_Type', 'Passenger_Count', 'Product_Category']
    X = preprocessor.transform(df_analisis[features])
    
    # Predecir
    predicciones = model.predict(X).flatten()
    
    # Crear columnas de desperdicio
    df_analisis['Predicted_Consumption'] = predicciones
    df_analisis['Predicted_Waste'] = df_analisis['Standard_Specification_Qty'] - df_analisis['Predicted_Consumption']
    df_analisis['Predicted_Waste'] = df_analisis['Predicted_Waste'].clip(lower=0)
    df_analisis['Waste_Percentage'] = (df_analisis['Predicted_Waste'] / df_analisis['Standard_Specification_Qty'] * 100).round(2)
    
    # Tomar solo los primeros 3 productos para la tabla
    report = df_analisis[['Product_Name', 'Predicted_Waste', 'Waste_Percentage']].head(10).to_dict(orient='records')
    
    return render_template("index.html", report=report)

if __name__ == "__main__":
    app.run(debug=True)
