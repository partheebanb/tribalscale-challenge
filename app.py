from flask import Flask, request, jsonify
from intelligent_text import IntelligentText

app = Flask(__name__)
service = IntelligentText()

@app.route("/summarize", methods=["POST"])
def summarize():
    body = request.get_json()
    if not body or "text" not in body:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = service.summarize(body["text"])
    return jsonify(result.model_dump())

if __name__ == "__main__":
    app.run(debug=True)
