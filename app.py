import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
from datetime import datetime
import socket
import sys
import pandas as pd
import numpy as np
import glob
from werkzeug.utils import secure_filename
from scipy.stats import ttest_rel


# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_FILES'] = 5  # Maximum number of files to keep

def rotate_files(directory):
    """Keep only the last MAX_FILES files in the directory"""
    files = sorted(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)
    while len(files) > app.config['MAX_FILES']:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file)
        except OSError as e:
            app.logger.error(f"Error deleting file {oldest_file}: {e}")
            
# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    local_ip = "127.0.0.1"  # Default to localhost
    try:
        # Try to get actual local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google DNS
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass  # Keep the default
    
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
        elif solver not in ['gurobi', 'glpk','highs']:
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
        # File rotation after successful processing
        rotate_files(app.config['UPLOAD_FOLDER'])
        rotate_files(app.config['OUTPUT_FOLDER'])

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
            model_type="AC" if model_type.lower() == "ac" else model_type.capitalize(),
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

@app.route('/error_analysis', methods=['GET', 'POST'])
def error_analysis():
    error_results = None
    
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        processed_files = []
        
        for file in uploaded_files:
            if file and file.filename.endswith('.xlsx'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Improved model type detection
                filename_lower = filename.lower()
                if 'ac' in filename_lower:
                    model_type = 'ac'
                elif 'btheta' in filename_lower:
                    model_type = 'btheta'
                elif 'bolognani' in filename_lower:
                    model_type = 'bolognani'
                elif 'decoupled' in filename_lower:
                    model_type = 'decoupled'
                else:
                    model_type = 'other'
                
                processed_files.append({
                    'filename': filename,
                    'model_type': model_type,
                    'filepath': filepath
                })
        
        ac_files = [f for f in processed_files if f['model_type'] == 'ac']
        other_files = [f for f in processed_files if f['model_type'] != 'ac']
        
        if not ac_files or not other_files:
            return render_template('error_analysis.html', 
                                error="You need to upload at least one AC file and one other model file")
        
        error_results = calculate_errors(ac_files[0], other_files)
    
    return render_template('error_analysis.html',
                         error_results=error_results)




def calculate_errors(ac_file, other_files):
    error_results = {}
    try:
        # Load AC data
        ac_data_results = pd.read_excel(ac_file['filepath'], sheet_name='Results')
        ac_data_production = pd.read_excel(ac_file['filepath'], sheet_name='Production')
        ac_data_reactive = pd.read_excel(ac_file['filepath'], sheet_name='Reactive')
        
        for file in other_files:
            model_name = file['model_type'].capitalize()
            model_data_results = pd.read_excel(file['filepath'], sheet_name='Results')
            
            errors = {}
            
            # Voltage calculations
            merged = pd.merge(model_data_results, ac_data_results, on='Bus', suffixes=('_model', '_ac'))
            
            # Voltage magnitude
            vm_abs_error = abs(merged['vm_pu_model'] - merged['vm_pu_ac'])
            errors.update({
                'vm_max_abs': f"{vm_abs_error.max():.6f} p.u.",
                'vm_mean_abs': f"{vm_abs_error.mean():.6f} p.u.",
            })
            
            # Voltage angle
            va_diff = merged['va_degrees_model'] - merged['va_degrees_ac']
            va_abs_error = np.minimum(abs(va_diff), 360 - abs(va_diff))
            errors.update({
                'va_max_abs': f"{va_abs_error.max():.2f}°",
                'va_mean_abs': f"{va_abs_error.mean():.2f}°",
            })
            
            # Active power calculations
            try:
                model_data_production = pd.read_excel(file['filepath'], sheet_name='Production')
                merged_p = pd.merge(model_data_production, ac_data_production, on='Bus', suffixes=('_model', '_ac'))
                
                p_abs_error = abs(merged_p['p_pu_model'] - merged_p['p_pu_ac'])
                errors.update({
                    'p_max_abs': f"{p_abs_error.max():.6f} p.u.",
                    'p_mean_abs': f"{p_abs_error.mean():.6f} p.u.",
                })
            except Exception as e:
                errors.update({
                    'p_max_abs': "N/A",
                    'p_mean_abs': "N/A",
                })
            
            # Reactive power calculations (skip for BTheta)
            if model_name.lower() != 'btheta':
                try:
                    model_data_reactive = pd.read_excel(file['filepath'], sheet_name='Reactive')
                    merged_q = pd.merge(model_data_reactive, ac_data_reactive, on='Bus', suffixes=('_model', '_ac'))
                    
                    q_abs_error = abs(merged_q['q_pu_model'] - merged_q['q_pu_ac'])
                    errors.update({
                        'q_max_abs': f"{q_abs_error.max():.6f} p.u.",
                        'q_mean_abs': f"{q_abs_error.mean():.6f} p.u.",
                    })
                except Exception as e:
                    errors.update({
                        'q_max_abs': "N/A",
                        'q_mean_abs': "N/A",
                    })
            else:
                errors.update({
                    'q_max_abs': "Not calculated",
                    'q_mean_abs': "Not calculated",
                })
            
            # Error distribution metrics
            errors.update({
                'error_95th_percentile': f"{np.percentile(vm_abs_error, 95):.6f} p.u.",
                'error_iqr': f"{vm_abs_error.quantile(0.75) - vm_abs_error.quantile(0.25):.6f} p.u.",
            })
            
            # Worst performing buses
            max_vm_error_idx = vm_abs_error.idxmax()
            max_va_error_idx = va_abs_error.idxmax()
            
            errors.update({
                'max_error_bus': str(merged.loc[max_vm_error_idx, 'Bus']),
                'max_error_value': f"{vm_abs_error.max():.6f} p.u.",
                'max_angle_bus': str(merged.loc[max_va_error_idx, 'Bus']),
                'max_angle_value': f"{va_abs_error.max():.2f}°",
            })
            
            error_results[model_name] = errors
            
    except Exception as e:
        app.logger.error(f"Error in calculate_errors: {str(e)}")
        return {"error": f"Calculation error: {str(e)}"}
    
    return error_results





@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)