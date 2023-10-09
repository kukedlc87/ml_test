import joblib
from flask import Flask, jsonify, request
import numpy as np


app = Flask(__name__)
@app.route("/")
def home():
    return 'La pagina esta funcionando'

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json(force=True)
    variables = np.array(data["Variables"]).reshape(1, -1)
    predictor = joblib.load('modelo.pkl')
    prediccion = predictor.predict_proba(variables)
    # Devuelve la respuesta en formato JSON
    return jsonify({"La probabilidad de default positivo es de: ": prediccion.tolist()})




if __name__ == '__main__':
    app.run()