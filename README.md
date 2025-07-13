# Optimal Power Flow Platform


## About This Project

This web application provides a **user-friendly interface** for running the Optimal Power Flow models developed in the original research repository:  
[Linear Approximation OPF Repository](https://github.com/ManousosAlexandrakis/Linear_Approximation_OPF.git)

Key improvements over the original implementation:
- **Web-based GUI** - No command-line usage required
- **Built-in visualization** - Immediate graphical results
- **Python implementation** - Easier deployment than Julia
- **All models in one place** - Unified interface for all three linear formulations
- **Simplified data handling** - Direct Excel file support

While the original repository contains the theoretical foundations and Julia implementations, this application offers:
- Enhanced usability for non-programmers
- Comparison/Plotting tools across different models


## Key Features

- **Model Execution**:
  - B-Theta linear model
  - Bolognani linear approximation
  - Decoupled linear model
- **Data Handling**:
  - Excel file input support
  - Results visualization
  - Downloadable outputs
- **Educational Resources**:
  - Mathematical formulation documentation


For the original Julia implementations and theoretical background, visit:  
[Linear Approximation OPF Repository](https://github.com/ManousosAlexandrakis/Linear_Approximation_OPF.git)

## Installation Guide

### Prerequisites

- Python 3.8+
- Required solvers:
  - Gurobi (recommended)
  - IPOPT (alternative)

### Quick Setup

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Project Structure
```
opf-platform/
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
├── run_BTheta_model.py        # B-Theta model implementation
├── run_Bolognani_model.py     # Bolognani model implementation
├── run_Decoupled_model.py     # Decoupled model implementation
├── uploads/                   # User uploads storage
├── output/                    # Generated results
├── static/                    # Static assets (JS/CSS/images)
│   ├── plotting.js            # Visualization logic
│   └── style.css              # Application styles
└── templates/                 # HTML templates
    ├── index.html             # Main interface
    ├── plotting.html          # Results visualization
    ├── result.html            # Results display
    └── theory.html            # Mathematical documentation
```

---





 ### 2. Run the Application

In the project folder, run the app.py file:

Then open your browser and go to:

http://127.0.0.1:5000

To close the app press Ctlr + C in python terminal.

