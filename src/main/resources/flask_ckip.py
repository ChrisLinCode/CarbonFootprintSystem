import os
from flask import Flask, request, jsonify
import unicodedata
from ckip_transformers.nlp import CkipWordSegmenter
import re

app = Flask(__name__)

# 初始化 CKIP Transformer 分詞器（只在服務啟動時初始化一次）
ws_driver = CkipWordSegmenter(model="bert-base")

def fix_text(text):
    # 根據需要進行 Unicode normalization
    return unicodedata.normalize("NFKC", text)

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    fixed_texts = [fix_text(t) for t in texts]
    segmented_result = ws_driver(fixed_texts)
    # 將分詞結果轉換成以空格分隔的字串列表
    result = [" ".join(words) for words in segmented_result]
    # 新增：將多個連續空格替換成一個空格，並去除前後空白
    result = [re.sub(r'\s+', ' ', seg_text).strip() for seg_text in result]
    return jsonify({"segmented": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))
