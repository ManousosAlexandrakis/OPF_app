import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
from datetime import datetime
import socket
import shutil
import sys


#julia_executable_path = 'julia'  # Ensure Julia is in your PATH or provide the full path to the executable
# Full path example
# julia_executable_path = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Julia-1.11.5\\bin\\julia.exe'
julia_executable_path = shutil.which("julia")

if julia_executable_path is None:
    print("Julia executable not found in PATH.")
else:
    print(f"Julia executable found at: {julia_executable_path}")
    
def find_julia_in_path():
    if julia_executable_path is None:
        print("Please specify the full path to your Julia executable manually in the script.")
        print("1. Press Windows Key ")
        print("2. Type: julia")
        print("3. Right-click on the Julia app → Click 'Open file location'")
        print("4. In the Explorer window, right-click the shortcut → 'Open file location' again.")
        print("5. Now you're in the folder where julia.exe lives. The path should look like this:")
        print("   `C:\\Users\\<YourName>\\AppData\\Local\\Programs\\Julia-1.11.5\\bin\\julia.exe`\n")
        print("6. Then update the script like this:")
        print("  julia_executable_path = r'C:\\Path\\To\\Julia\\bin\\julia.exe'  # for Windows")
        print("  julia_executable_path = '/usr/local/bin/julia'  # for macOS/Linux\n")
        sys.exit("")

find_julia_in_path()
    
    
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
        
        if not file or not file.filename.endswith('.xlsx'):
            return redirect(url_for('index'))
        
        if model_type not in ['bolognani', 'btheta', 'decoupled', 'ac']:
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
            'decoupled': 'run_Decoupled_model.jl',
            'ac': 'run_AC_model.jl'
        }

        # Run the Julia model
        result = subprocess.run(
            [julia_executable_path, model_scripts[model_type], input_path, output_path],
            capture_output=True,
            text=True
        )

        # Extract solver status from Julia output
        status_line = next(
    (line for line in result.stdout.splitlines() if "status:" in line.lower()),
    None
)
        status = "Unknown"  # Default if nothing matches

        if status_line:
            status_upper = status_line.upper()
            if "OPTIMAL" in status_upper:
                status = "Optimal"
            elif "INFEASIBLE" in status_upper:
                status = "Infeasible"
            elif "LOCALLY_SOLVED" in status_upper:
                status = "Locally Solved"
            elif "UNBOUNDED" in status_upper:
                status = "Unbounded"
            elif "TIME_LIMIT" in status_upper:
                status = "Time Limit Reached"
            elif "ITERATION_LIMIT" in status_upper:
                status = "Iteration Limit Reached"
            elif "MEMORY_LIMIT" in status_upper:
                status = "Memory Limit Reached"
            elif "OTHER_ERROR" in status_upper:
                status = "Other Error"
            elif "COMPUTATION_ERROR" in status_upper:
                status = "Computation Error"
            elif "INTERRUPTED" in status_upper:
                status = "Interrupted"
            elif "USER_LIMIT" in status_upper:
                status = "Stopped by User Limit"
            elif "NUMERICAL_ERROR" in status_upper:
                status = "Numerical Error"
            elif "INVALID_MODEL" in status_upper:
                status = "Invalid Model"
            else:
                status = status_line.strip()  # fallback: keep raw line

        if result.returncode != 0:
            error_msg = f"Julia script failed:\n{result.stderr}"
            return error_msg, 500

        if not os.path.exists(output_path):
            return "Output file was not created", 500

        return render_template(
            "result.html",
            output_filename=output_filename,
            model_type=model_type.capitalize(),
            status=status
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