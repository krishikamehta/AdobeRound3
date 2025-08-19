import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from core_logic import process_documents_for_web

MODEL_PATH = os.environ.get("MODEL_PATH", "models/all-MiniLM-L6-v2")

app = Flask(__name__, static_folder='/var/www/html')
CORS(app)

ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_documents():
    if 'files' not in request.files or 'metadata' not in request.form:
        return jsonify({"error": "Missing files or metadata"}), 400

    metadata_str = request.form['metadata']
    try:
        metadata = json.loads(metadata_str)
        persona = metadata.get('persona', 'Default')
        job_to_be_done = metadata.get('job_to_be_done', 'Analyze documents.')
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON for metadata"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        input_documents_list = []
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        for file in files:
            if file and allowed_file(file.filename):
                rel_path = file.filename
                full_path = os.path.join(temp_dir, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                file.save(full_path)
                input_documents_list.append({"filename": rel_path})

        mock_input = {
            "documents": input_documents_list,
            "persona": {"role": persona},
            "job_to_be_done": {"task": job_to_be_done}
        }
        
        input_json_path = os.path.join(temp_dir, 'challenge1b_input.json')
        with open(input_json_path, 'w') as f:
            json.dump(mock_input, f, indent=4)
        
        try:
            output_data = process_documents_for_web(temp_dir, temp_dir, MODEL_PATH)
            return jsonify(output_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # This is for local development. In Docker, Gunicorn handles the server.
    app.run(host='127.0.0.1', port=8000)
