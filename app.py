import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
from datetime import datetime
import socket
import sys
import pandas as pd

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # Get Local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return render_template('index.html', local_ip=local_ip)

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form data
        file = request.files.get('file')
        model_type = request.form.get('model')
        solver = request.form.get('solver', 'gurobi')  # Default to Gurobi if not specified
        
        if not file or not file.filename.endswith('.xlsx'):
            return redirect(url_for('index'))
        
        if model_type not in ['bolognani', 'btheta', 'decoupled', 'ac']:
            return "Invalid model selected", 400

        # Handle solver selection
        if model_type == 'ac':
            solver = 'ipopt'  # Force IPOPT for AC model
        elif solver not in ['gurobi', 'glpk']:
            solver = 'gurobi'  # Default to Gurobi for linear models

        # Save input file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"input_{timestamp}.xlsx"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        # Prepare output
        output_filename = f"results_{model_type}_{solver}_{timestamp}.xlsx"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Map model types to scripts
        model_scripts = {
            'btheta': 'run_BTheta_model.py',
            'bolognani': 'run_Bolognani_model.py',
            'decoupled': 'run_Decoupled_model.py',
            'ac': 'run_AC_model.py'
        }

        if model_type not in model_scripts:
            return "Model not yet implemented in Python", 400

        # Prepare environment with solver selection
        env = os.environ.copy()
        env['SOLVER'] = solver

        # Run the Python model with the selected solver
        result = subprocess.run(
            [sys.executable, model_scripts[model_type], input_path, output_path],
            capture_output=True,
            text=True,
            env=env
        )

        solver_status_line = next(
            (line for line in result.stdout.splitlines() if "solver_status:" in line.lower()),
            None
        )
        termination_line = next(
            (line for line in result.stdout.splitlines() if "termination_condition:" in line.lower()),
            None
        )

        # Extract solver status and termination condition from script output
        solver_status_line = next(
            (line for line in result.stdout.splitlines() if "status:" in line.lower()),
            None
        )
        termination_line = next(
            (line for line in result.stdout.splitlines() if "termination_condition:" in line.lower()),
            None
        )

        solver_status = "Unknown"
        termination_condition = "Unknown"

        if solver_status_line:
            solver_status = solver_status_line.split(":", 1)[-1].strip()

        if termination_line:
            termination_condition = termination_line.split(":", 1)[-1].strip()

        # Check if output file exists before trying to load it
        if not os.path.exists(output_path):
            return render_template(
                "result.html",
                output_filename=output_filename,
                model_type=model_type.capitalize(),
                solver=solver,
                solver_status=solver_status,
                termination_condition=termination_condition,
                error="Output file was not created."
            )

        # Read the Excel file to pass data to template
        excel_data = pd.ExcelFile(output_path)

        # Prepare preview data for each sheet
        preview_data = {
            'results_data': excel_data.parse('Results').head(100).to_dict('records'),
            'production_data': excel_data.parse('Production').head(100).to_dict('records'),
            'price_data': excel_data.parse('LMP').head(100).to_dict('records'),
            'flows_data': excel_data.parse('Flows').head(100).to_dict('records')
        }

        # Add reactive data if sheet exists and model supports it
        if model_type.lower() in ['bolognani', 'decoupled', 'ac']:
            try:
                preview_data['reactive_data'] = excel_data.parse('Reactive').head(100).to_dict('records')
            except ValueError:
                preview_data['reactive_data'] = []  # Sheet not found

        # Final render
        return render_template(
            "result.html",
            output_filename=output_filename,
            model_type=model_type.capitalize(),
            solver=solver,
            solver_status=solver_status,
            termination_condition=termination_condition,
            **preview_data
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

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# if __name__ == '__main__':
#     # Let Streamlit handle the server instead
#     pass  # Or add Streamlit-specific logic