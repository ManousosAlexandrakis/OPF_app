# Optimal Power Flow Platform

This is a web-based application designed to run **three linear models** of the Optimal Power Flow (OPF) problem. The app allows users to:

- Upload input data files (e.g., Excel format)
- Run different linear OPF models implemented in Julia
- View and download plots of results
- Explore the underlying mathematical formulation of each model

## Project Structure

 - app.py                # Main Flask application
 
- templates/            # HTML templates 

- static/               # CSS, JS, images

- uploads/              # Folder for user-uploaded files

- output/               # Folder for generated results

- run_BTheta_model.jl   # Julia script for B-Theta model

- run_Bolognani_model.jl # Julia script for Bolognani model

- run_Decoupled_model.jl # Julia script for Decoupled model

## Getting Started

In the app.py file you have to write the executable path of Julia.

```python
julia_executable_path = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Julia-1.11.5\\bin\\julia.exe'
```

To find the path to julia executable you can run the following command to a command prompt:

```bash
where /r C:\ julia.exe
```

---

### 1. Python and Julia Setup

Make sure you have Python 3.8+ installed.


#### Clone the Repository

```bash```

```bash
git clone https://github.com/ManousosAlexandrakis/OPF_app.git
cd OPF_app
```



#### Install required Python packages:

Make sure you have Python 3.8+ installed.

```bash
pip install flask
```


#### Install required Julia packages:
Make sure you have Julia installed (version 1.8 or later recommended).

Start Julia in the terminal:

Then in the Julia REPL, enter package mode (press ]) and run:
```julia
add JuMP Gurobi XLSX DataFrames
```

 ### 2. Run the Application

In the project folder, start the Flask app:

Then open your browser and go to:

http://127.0.0.1:5000

To close the app press ctlr + C

