from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Simple Flask Test', 
        'status': 'working',
        'wsl_ip': '172.29.37.89',
        'port': 5001
    })

@app.route('/test')
def test():
    return '<h1>Flask Test Working!</h1>'

if __name__ == '__main__':
    print("🚀 Starting simple Flask test server...")
    print("📡 Test URLs:")
    print("   http://127.0.0.1:5001")
    print("   http://172.29.37.89:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
