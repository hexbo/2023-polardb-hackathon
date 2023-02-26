# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from model import Model
from utils import  calcEmbedding, fetchResultByEmbedding

app = Flask(__name__)
model = Model("./model.pb")

@app.route("/api/query-by-embedding", methods=["POST"])
def queryByEmbedding():
    if request.json == None: return jsonify({"status": "error"})

    data = request.json
    parsedResult = fetchResultByEmbedding(data)
    return jsonify({ "status": "success", "data": parsedResult })


@app.route("/api/predict", methods=["POST"])
def predict():
    if request.json == None: return jsonify({"status": "error"})

    jsonData = request.json
    result = calcEmbedding(model, jsonData)
    return jsonify({ "status": "success", "data": result })


if __name__ == '__main__':
    app.run('0.0.0.0')