from pyomo.environ import *
import pandas as pd
import numpy as np
import sys
from math import degrees as rad2deg
import os


def run_btheta_model(input_path, output_path):
    """Run BTheta model with input from Excel and save results to Excel"""
    
    try:
        # Get solver from environment variable (default to gurobi)
        solver_name = os.getenv('SOLVER', 'gurobi')
        
        # Set solver
        solver = SolverFactory("gurobi")
        # Set solver based on selection
        if solver_name == 'gurobi':
            solver = SolverFactory("gurobi")
        elif solver_name == 'glpk':
            solver = SolverFactory("glpk")
        else:
            raise ValueError(f"Unsupported solver for BTheta model: {solver_name}")

        ###### Data Handling ######
        def load_excel(filepath, sheet_name, fill_empty_values=True):
            output_dataframe = pd.read_excel(filepath, sheet_name=sheet_name)
            if fill_empty_values:
                output_dataframe = output_dataframe.fillna(0)
            return output_dataframe

        # Load all required sheets from input Excel
        try:
            sgen_data = load_excel(input_path, "gen")
            Edges = load_excel(input_path, "edges")
            bus_data = load_excel(input_path, "bus")
            load_data = load_excel(input_path, "load")
            slack_data = load_excel(input_path, "ext_grid")
            Upward_data = load_excel(input_path, "Upward")
        except Exception as e:
            print(f"Error loading Excel sheets: {str(e)}")
            return "Error loading input data", False

        # Data for slack bus
        try:
            slack_v = slack_data["vm_pu"].iloc[0]      # 1
            slack_degree = slack_data["va_degree"].iloc[0] # 0
            slack_bus = slack_data["bus"].iloc[0] # 1000
        except Exception as e:
            print(f"Error processing slack bus data: {str(e)}")
            return "Error processing slack bus data", False

        buses = bus_data["bus"].tolist()
        edges_index = Edges.index.tolist()

        # Create a dictionary to store the index of each bus
        bus_id_to_index = {buses[i]: i for i in range(len(buses)-1)}
        bus_id_to_index[slack_bus] = buses.index(slack_bus)

        # Create a dictionary mapping edges to FlowMax
        Flowmax_dict = dict()
        for i in range(len(Edges)): 
            to_bus = Edges["to_bus"].iloc[i]
            from_bus = Edges["from_bus"].iloc[i]
            Flowmax_dict[(from_bus, to_bus)] = Edges["FlowMax"].iloc[i]
            Flowmax_dict[(to_bus, from_bus)] = Edges["FlowMax"].iloc[i]
            
        Flowmax_edge_dict = dict()
        for i in range(len(Edges)):
            index = Edges["idx"].iloc[i] - 1
            Flowmax_edge_dict[i] = Edges["FlowMax"].iloc[index]
            
        # Create a dictionary to store connected buses
        connected_buses_dict = dict()

        for i in range(len(Edges)):
            from_bus = int(Edges["from_bus"].iloc[i])
            to_bus = int(Edges["to_bus"].iloc[i])

            # Add to_bus to from_bus's connection list
            if from_bus not in connected_buses_dict:
                connected_buses_dict[from_bus] = set()
            connected_buses_dict[from_bus].add(to_bus)

            # Add from_bus to to_bus's connection list
            if to_bus not in connected_buses_dict:
                connected_buses_dict[to_bus] = set()
            connected_buses_dict[to_bus].add(from_bus)

        # Convert sets to sorted lists
        for bus in connected_buses_dict:
            connected_buses_dict[bus] = sorted(connected_buses_dict[bus])

        # Total number of buses and edges
        n = len(buses)
        Edges_leng = len(Edges)

        # Create dictionaries to store PU, MinQ and MaxQ parameters for upward generators
        PU = {Upward_data["Bus"].iloc[i] : Upward_data["PU"].iloc[i] for i in range(len(Upward_data))}
        MinQ = {Upward_data["Bus"].iloc[i] : Upward_data["MinQ"].iloc[i] for i in range(len(Upward_data))}
        MaxQ = {Upward_data["Bus"].iloc[i] : Upward_data["MaxQ"].iloc[i] for i in range(len(Upward_data))}

        Upward_set = set(Upward_data["Bus"])
        buses_except_upward = set(buses) - set(Upward_set)

        # Create Y matrix
        y = np.zeros((n,n), dtype=complex)

        for i in range(len(Edges)):
            From_bus = bus_id_to_index[Edges["from_bus"].iloc[i]]
            To_bus = bus_id_to_index[Edges["to_bus"].iloc[i]]
            r = 0
            x = Edges["X_pu"].iloc[i]
            z = complex(r,x)
            y[From_bus, To_bus] = 1/z
            y[To_bus, From_bus] = 1/z # symmetric
            
        Y = np.zeros((n,n), dtype=complex)

        for k in buses:
            Y[bus_id_to_index[k], bus_id_to_index[k]] = sum(y[bus_id_to_index[k],bus_id_to_index[m]] for m in connected_buses_dict[k])

        for k in buses:
            for m in connected_buses_dict[k]:
                Y[bus_id_to_index[k], bus_id_to_index[m]] = -y[bus_id_to_index[k], bus_id_to_index[m]]

        B = Y.imag

        Load_set = set(load_data["bus"])
        buses_without_load = set(buses) - set(Load_set)

        Load_dict = {load_data["bus"].iloc[i] : load_data["p_mw"].iloc[i] for i in range(len(load_data))}

        ##### Math Optimization model ######

        # Model
        model = ConcreteModel(name="BTheta")

        # Variables
        model.p = Var(buses, within=NonNegativeReals)
        model.f = Var(range(n), range(n), within=Reals)
        model.delta = Var(buses, within=Reals)
        model.u = Var(buses, within=Reals)
        model.generation_cost = Var()

        # CONSTRAINTS

        # Real Power Flow calculation for each edge
        def real_power_flow(model,m,n):
            return model.f[bus_id_to_index[m],bus_id_to_index[n]] == B[bus_id_to_index[m], bus_id_to_index[n]] *(model.delta[m] - model.delta[n])
        connected_bus_pairs = [(m, n) for m in buses for n in connected_buses_dict[m]]
        model.real_power_flow = Constraint(connected_bus_pairs, rule=real_power_flow)

        # Voltage magnitude for all buses is considered 1 p.u.
        def voltage_magnitude(model, m):
            return model.u[m] == 1
        model.voltage_magnitude = Constraint(buses, rule=voltage_magnitude)

        def slack_bus_delta(model):
            return model.delta[slack_bus] == 0
        model.slack_bus_delta = Constraint(rule=slack_bus_delta)

        def power_generation_upper_limit(model, m):
            if m in Upward_set:
                return model.p[m] <= MaxQ[m]
            else:
                return model.p[m] == 0
        model.power_generation_upper_limit = Constraint(buses, rule=power_generation_upper_limit)

        def power_generation_lower_limit(model, m):
            if m in Upward_set:
                return model.p[m] >= MinQ[m]
            else:
                return model.p[m] == 0
        model.power_generation_lower_limit = Constraint(buses, rule=power_generation_lower_limit)
            
        def flows_limit(model,m,n):
            return model.f[bus_id_to_index[m], bus_id_to_index[n]] <= Flowmax_dict[(m,n)]
        model.flows_limit = Constraint(connected_bus_pairs, rule=flows_limit)

        def nodal_power_balance(model,m):
            return -sum(model.f[bus_id_to_index[m], bus_id_to_index[n]] for n in connected_buses_dict[m]) + model.p[m] - Load_dict.get(m,0) == 0
        model.nodal_power_balance = Constraint(buses, rule=nodal_power_balance)
        
        # Objective Function
        def obj_rule(model):
            return model.generation_cost
        model.obj_rule = Objective(rule=obj_rule,sense=minimize)

        def objective_function(model):
            return model.generation_cost == sum(model.p[m] * PU[m] for m in Upward_set)
        model.objective_function = Constraint(rule=objective_function)

        # Set up dual variables
        model.dual = Suffix(direction=Suffix.IMPORT)

        # Set solver
        results = solver.solve(model, tee=True)


        
        # Check solution status
        if str(results.solver.status) != "ok":
            return f"Solver failed with status: {results.solver.status}", False
        if str(results.solver.termination_condition) not in ["optimal", "locallyOptimal"]:
            return f"Solver terminated with condition: {results.solver.termination_condition}", False

        # Create DataFrames for results
        prod_df = pd.DataFrame({
            "Bus": list(Upward_set),
            "p_pu": [value(model.p[i]) for i in list(Upward_set)],
            "pmax_pu": [MaxQ[i] for i in list(Upward_set)],
            "pmin_pu": [MinQ[i] for i in list(Upward_set)],
            "PU_euro/MWh": [PU[i] for i in Upward_set]
        })

        results_df = pd.DataFrame({
            "Bus": buses,
            "vm_pu": [value(model.u[i]) for i in buses],
            "va_degrees": [rad2deg(value(model.delta[i])) for i in buses]
        })

        price_df = pd.DataFrame({
            "Bus": buses,
            "nodal_price_euro/MWh": [model.dual[model.nodal_power_balance[i]] for i in buses]
        })

        # flows_df = pd.DataFrame({
        #     "Edge": [i + 1 for i in edges_index],
        #     "from_bus": [Edges["from_bus"].iloc[i] for i in range(Edges_leng)],
        #     "flows_to": [Edges["to_bus"].iloc[i] for i in range(Edges_leng)],
        #     "Flow_p_pu": [value(model.f[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
        #     "Flowmax_pu": [Flowmax_edge_dict[i] for i in range(len(Edges))]
        # })
        
        flows_df = pd.DataFrame({
            "Edge": [i + 1 for i in edges_index],
            "from_bus": [Edges["from_bus"].iloc[i] for i in range(Edges_leng)],
            "flows_to": [Edges["to_bus"].iloc[i] for i in range(Edges_leng)],
            "Flows_p_pu_from": [value(model.f[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flows_p_pu_to": [value(model.f[bus_id_to_index[Edges["to_bus"].iloc[i]], bus_id_to_index[Edges["from_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flowmax_pu": [Flowmax_edge_dict[i] for i in range(len(Edges))]
        })

        # Save results to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            prod_df.to_excel(writer, sheet_name="Production", index=False)
            price_df.to_excel(writer, sheet_name="LMP", index=False)
            flows_df.to_excel(writer, sheet_name="Flows", index=False)
            


        return "status: Optimal", True

    except Exception as e:
        print(f"Error in BTheta model: {str(e)}")
        return f"Error: {str(e)}", False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_BTheta_model.py <input_path> <output_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    status, success = run_btheta_model(input_path, output_path)
    print(status)
    
    if not success:
        sys.exit(1)