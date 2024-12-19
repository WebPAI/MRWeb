from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
@app.route('/backend', methods=['POST'])
def index():
    data = request.json
    print("!!!!!!!!data received!!!!!!!")
    print(data)
    return jsonify(data)
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)