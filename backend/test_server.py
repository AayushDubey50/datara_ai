from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return jsonify({"message": "Test server running", "status": "ok"})

if __name__ == "__main__":
    print("Starting simple Flask server on port 5000...")
    app.run(host="127.0.0.1", port=5000, debug=False)
