import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
from datetime import datetime


julia_executable_path = 'julia'  # Ensure Julia is in your PATH or provide the full path to the executable
# Full path example
# julia_executable_path = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Julia-1.11.5\\bin\\julia.exe'
# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form data
        file = request.files.get('file')
        model_type = request.form.get('model')
        
        if not file or not file.filename.endswith('.xlsx'):
            return redirect(url_for('index'))
        
        if model_type not in ['bolognani', 'btheta', 'decoupled']:
            return "Invalid model selected", 400

        # Save input file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"input_{timestamp}.xlsx"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        # Prepare output
        output_filename = f"results_{model_type}_{timestamp}.xlsx"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Map model types to Julia scripts
        model_scripts = {
            'bolognani': 'run_Bolognani_model.jl',
            'btheta': 'run_BTheta_model.jl',
            'decoupled': 'run_Decoupled_model.jl'
        }

        # Run the Julia model
        result = subprocess.run(
            [julia_executable_path, model_scripts[model_type], input_path, output_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Julia script failed:\n{result.stderr}"
            return error_msg, 500

        if not os.path.exists(output_path):
            return "Output file was not created", 500

        return render_template(
            "result.html",
            output_filename=output_filename,
            model_type=model_type.capitalize()
        )

    except Exception as e:
        app.logger.error(f"Error in submit: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            as_attachment=True
        )
    except FileNotFoundError:
        return "File not found", 404
    
@app.route('/theory')
def theory():
    """Show mathematical models documentation"""
    return render_template("theory.html")

@app.route('/plotting')
def plotting():
    return render_template("plotting.html")

# Add this to serve static files (if not already present)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)