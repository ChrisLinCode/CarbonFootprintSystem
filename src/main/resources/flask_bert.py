#!/usr/bin/env python3
# coding: utf-8

import os
from flask import Flask, request, jsonify
import io, sys, unicodedata
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# 強制設定 stdout 為 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 載入 BERT 模型
model = SentenceTransformer("uer/sbert-base-chinese-nli")

def fix_text(text):
    return unicodedata.normalize("NFKC", text)

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    fixed_texts = [fix_text(t) for t in texts]
    embeddings = model.encode(fixed_texts)
    # 將每筆 embedding 轉為 list
    embeddings_list = [emb.tolist() for emb in embeddings]
    return jsonify({"embeddings": embeddings_list})

if __name__ == '__main__':
    # 建議部署時採用 Gunicorn 或 Docker，此處僅用於測試
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
