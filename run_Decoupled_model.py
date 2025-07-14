import pandas as pd
import numpy as np
from pandas import DataFrame
from math import degrees as rad2deg 
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import sys, os

def run_decoupled_model(input_path, output_path):
    """Run Decoupled model with input from Excel and save results to Excel"""    
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
            raise ValueError(f"Unsupported solver for Decoupled model: {solver_name}")
        ###### Data Handling ######
        def load_excel(filepath,sheet_name, fill_empty_values = True):
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
            slack_v = slack_data["vm_pu"].iloc[0]
            slack_delta = slack_data["va_degree"].iloc[0]
            slack_bus = slack_data["bus"].iloc[0]
        except Exception as e:
            print(f"Error processing slack bus data: {str(e)}")
            return "Error processing slack bus data", False

        #
        buses = bus_data["bus"].tolist()
        edges_index = Edges.index.tolist()
        # edges_index = Edges["idx"].tolist()

        # Create a dictionary to store the index of each bus
        # slack_index = buses.index(slack_bus)

        bus_id_to_index = {buses[i]: i for i in range(len(buses)-1)}
        bus_id_to_index[slack_bus] = buses.index(slack_bus)



        # # Create a dictionary mapping edges to FlowMax
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
            
        # # Sbase of the system
        Ssystem = 1
        # # Create a dictionary to store connected buses
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
            #print(connected_buses_dict)

        # print 
        # for bus, neighbors in connected_buses_dict.items():
        #     print(f"Bus {bus} is connected to: {neighbors}")
            
            
        # Total number of buses and edges
        n = len(buses)
        Edges_leng =  len(Edges)

        # # Create dictionaries to store PU, MinQ and MaxQ parameters for upward generators
        PU = {Upward_data["Bus"].iloc[i] : Upward_data["PU"].iloc[i] for i in range(len(Upward_data))}
        # Minimum and Maximum active power for generators
        MinQ = {Upward_data["Bus"].iloc[i] : Upward_data["MinQ"].iloc[i]/Ssystem for i in range(len(Upward_data))}
        MaxQ = {Upward_data["Bus"].iloc[i] : Upward_data["MaxQ"].iloc[i]/Ssystem for i in range(len(Upward_data))}
        # Minimum and Maximum reactive power for generators
        Qmin = {sgen_data["bus"].iloc[i] : sgen_data["QRmin"].iloc[i]/Ssystem for i in range(len(sgen_data))}
        Qmax = {sgen_data["bus"].iloc[i] : sgen_data["QRmax"].iloc[i]/Ssystem for i in range(len(sgen_data))}

        Upward_set = set(Upward_data["Bus"])
        buses_except_upward = set(buses) - set(Upward_set)

        # # Create Y matrix
        y = np.zeros((n,n), dtype=complex)


        for i in range(len(Edges)):
            From_bus = bus_id_to_index[Edges["from_bus"].iloc[i]]
            To_bus = bus_id_to_index[Edges["to_bus"].iloc[i]]
            r = Edges["R_pu"].iloc[i]
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

        #print("Y matrix:", Y)

        B = Y.imag
        #print("B matrix:", B)

        Load_set = set(load_data["bus"])
        buses_without_load = set(buses) - set(Load_set)

        Load_dict = {load_data["bus"].iloc[i] : load_data["p_mw"].iloc[i]/Ssystem for i in range(len(load_data))}
        Load_dict_q = {load_data["bus"].iloc[i] : load_data["q_mvar"].iloc[i]/Ssystem for i in range(len(load_data))}




        ##### Math Optimization model ######

        # # Model
        model = ConcreteModel(name="Decoupled")

        # # Variables
        model.p = Var(buses, within=NonNegativeReals)
        model.q = Var(buses, within=Reals)
        model.f = Var(range(n), range(n), within=Reals)
        model.fq = Var(range(n), range(n), within=Reals)
        model.delta = Var(buses, within=Reals)
        model.u = Var(buses, within=Reals)

        model.generation_cost = Var()

        # # CONSTRAINTS

        #  # Real Power Flow calculation for each edge
        def active_power_flow(model,m,n):
            return model.f[bus_id_to_index[m],bus_id_to_index[n]] == B[bus_id_to_index[m], bus_id_to_index[n]] *(model.delta[m] - model.delta[n])
        connected_bus_pairs = [(m, n) for m in buses for n in connected_buses_dict[m]]
        model.active_power_flow = Constraint(connected_bus_pairs, rule=active_power_flow)


        def reactive_power_flow(model,m,n):
            return model.fq[bus_id_to_index[m],bus_id_to_index[n]] == B[bus_id_to_index[m], bus_id_to_index[n]] *(model.u[m] - model.u[n])
        connected_bus_pairs = [(m, n) for m in buses for n in connected_buses_dict[m]]
        model.reactive_power_flow = Constraint(connected_bus_pairs, rule=reactive_power_flow)

        # print("Real Power Flow Constraints:")
        # for key in model.active_power_flow:
        #     print(f"Constraint {key}: {model.active_power_flow[key].expr}")

        # # Voltage magnitude for all buses is considered 1 p.u.
        def voltage_magnitude_upper_limit(model, m):
            return model.u[m] <= 1.2
        model.voltage_magnitude_upper_limit = Constraint(buses, rule=voltage_magnitude_upper_limit)

        def voltage_magnitude_lower_limit(model, m):
            return model.u[m] >= 0.8
        model.voltage_magnitude_lower_limit = Constraint(buses, rule=voltage_magnitude_lower_limit)

        def voltage_magnitude_slack_bus(model):
            return model.u[slack_bus] == slack_v
        model.voltage_magnitude_slack_bus = Constraint(rule=voltage_magnitude_slack_bus)



        # print("Voltage Magnitude upper limits Constraints:")
        # for key in model.voltage_magnitude_upper_limit:
        #     print(f"Constraint {key}: {model.voltage_magnitude_upper_limit[key].expr}")

        def slack_bus_delta(model):
            return model.delta[slack_bus] == 0
        model.slack_bus_delta = Constraint(rule=slack_bus_delta)

        # print("Slack Bus Delta Constraint:")
        # print(f"Constraint: {model.slack_bus_delta.expr}]")

        def power_generation_upper_limit(model, m):
            if m in Upward_set:
                return model.p[m] <= MaxQ[m]
            else:
                return model.p[m] == 0
        model.power_generation_upper_limit = Constraint(buses, rule=power_generation_upper_limit)

        # print("Power Generation Upper Limit Constraints:")
        # for key in model.power_generation_upper_limit:
        #     print(f"Constraint {key}: {model.power_generation_upper_limit[key].expr}")

        def power_generation_lower_limit(model, m):
            if m in Upward_set:
                return model.p[m] >= MinQ[m]
            else:
                return model.p[m] == 0
        model.power_generation_lower_limit = Constraint(buses, rule=power_generation_lower_limit)

        # print("Power Generation Lower Limit Constraints:")
        # for key in model.power_generation_lower_limit:
        #     print(f"Constraint {key}: {model.power_generation_lower_limit[key].expr}")


        def reactive_power_generation_upper_limit(model, m):
            if m in Upward_set:
                return model.q[m] <= Qmax[m]
            else:
                return model.q[m] == 0
        model.reactive_power_generation_upper_limit = Constraint(buses, rule=reactive_power_generation_upper_limit)


        def reactive_power_generation_lower_limit(model, m):
            if m in Upward_set:
                return model.q[m] >= Qmin[m]
            else:
                return model.q[m] == 0
        model.reactive_power_generation_lower_limit = Constraint(buses, rule=reactive_power_generation_lower_limit)



            
        def active_flows_limit(model,m,n):
            return model.f[bus_id_to_index[m], bus_id_to_index[n]] <= Flowmax_dict[(m,n)]
        model.active_flows_limit = Constraint(connected_bus_pairs, rule=active_flows_limit)

        def reactive_flows_limit(model,m,n):
            return model.fq[bus_id_to_index[m], bus_id_to_index[n]] <= Flowmax_dict[(m,n)]
        model.reactive_flows_limit = Constraint(connected_bus_pairs, rule=reactive_flows_limit)

        # Print Constraints
        # print("Active Flows Limit Constraints:")
        # for key in model.active_flows_limit:
        #     print(f"Constraint {key}: {model.active_flows_limit[key].expr}")
            
        # print("Reactive Flows Limit Constraints:")
        # for key in model.reactive_flows_limit:
        #     print(f"Constraint {key}: {model.reactive_flows_limit[key].expr}")
            
            

        def nodal_power_balance(model,m):
            return -sum(model.f[bus_id_to_index[m], bus_id_to_index[n]] for n in connected_buses_dict[m]) + model.p[m] - Load_dict.get(m,0) == 0
        model.nodal_power_balance = Constraint(buses, rule=nodal_power_balance)

        def reactive_nodal_power_balance(model,m):
            return -sum(model.fq[bus_id_to_index[m], bus_id_to_index[n]] for n in connected_buses_dict[m]) + model.q[m] - Load_dict_q.get(m,0) == 0
        model.reactive_nodal_power_balance = Constraint(buses, rule=reactive_nodal_power_balance)

        # print('Active Nodal Power Balance Constraints:')
        # for key in model.nodal_power_balance:
        #     print(f"Constraint {key}: {model.nodal_power_balance[key].expr}")
            
        # print('Reactive Nodal Power Balance Constraints:')
        # for key in model.reactive_nodal_power_balance:
        #     print(f"Constraint {key}: {model.reactive_nodal_power_balance[key].expr}")
            
            
        # # Objective Function
        def obj_rule(model):
            return model.generation_cost
        model.obj_rule = Objective(rule=obj_rule,sense=minimize)

        def objective_function(model):
            return model.generation_cost == sum(model.p[m] * PU[m] for m in Upward_set)
        model.objective_function = Constraint(rule=objective_function)

        # Set up dual variables
        model.dual = Suffix(direction=Suffix.IMPORT)

        # Set solver
        solver = SolverFactory("gurobi")

        
        results = solver.solve(model, tee=True)
        
        # Check solution status
        if str(results.solver.status) != "ok":
            return f"Solver failed with status: {results.solver.status}", False
        if str(results.solver.termination_condition) not in ["optimal", "locallyOptimal"]:
            return f"Solver terminated with condition: {results.solver.termination_condition}", False        



        # Create a DataFrame for upward production
        prod_df = DataFrame({
            "Bus": list(Upward_set),
            "p_pu": [value(model.p[i]) for i in list(Upward_set)],
            "pmax_pu": [MaxQ[i] for i in list(Upward_set)],
            "pmin_pu":[MinQ[i] for i in list(Upward_set)],
            "PU_euro/MWh" : [PU[i] for i in list(Upward_set)]
        })

        Qreact_df = DataFrame({
            "Bus": list(Upward_set),
            "q_pu": [value(model.q[i]) for i in list(Upward_set)],
            "qmax_pu": [Qmax[i] for i in list(Upward_set)],
            "qmin_pu":[Qmin[i] for i in list(Upward_set)]
        })
        #print(">> Voltages at each bus:")

        # Print voltage results for each bus
        # for m in buses:
        #     print(f">> {m}: |V| = {value(model.u[m]):.1f} pu, angle = {rad2deg(value(model.delta[m])):.6f}°")

        # Create DataFrame for results
        results_df = DataFrame({
            "Bus": list(buses),
            "vm_pu": [value(model.u[i]) for i in buses],
            "va_degrees": [rad2deg(value(model.delta[i])) for i in buses]
        })

        price_df = DataFrame({
            "Bus": buses,
            "nodal_price_euro/MWh": [model.dual[model.nodal_power_balance[i]] for i in buses]
        })

        from_bus = Edges["from_bus"].tolist()
        to_bus = Edges["to_bus"].tolist()
        # flows_df = DataFrame({
        #     "Edge": [i + 1 for i in edges_index],
        #     "from_bus": [Edges["from_bus"].iloc[i] for i in range(Edges_leng)],
        #     "flows_to": [Edges["to_bus"].iloc[i] for i in range(Edges_leng)],
        #     "Flow_p_pu": [value(model.f[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
        #     "Flow_q_pu": [value(model.fq[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
        #     "Flowmax_pu": [Flowmax_edge_dict[i] for i in range(len(Edges))] 
        # })
        flows_df = DataFrame({
            "Edge": [i + 1 for i in edges_index],
            "from_bus": [Edges["from_bus"].iloc[i] for i in range(Edges_leng)],
            "flows_to": [Edges["to_bus"].iloc[i] for i in range(Edges_leng)],
            "Flows_p_pu_from": [value(model.f[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flows_p_pu_to": [value(model.f[bus_id_to_index[Edges["to_bus"].iloc[i]], bus_id_to_index[Edges["from_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flows_q_pu_from": [value(model.fq[bus_id_to_index[Edges["from_bus"].iloc[i]], bus_id_to_index[Edges["to_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flows_q_pu_to": [value(model.fq[bus_id_to_index[Edges["to_bus"].iloc[i]], bus_id_to_index[Edges["from_bus"].iloc[i]]]) for i in range(Edges_leng)],
            "Flowmax_pu": [Flowmax_edge_dict[i] for i in range(len(Edges))]
        })


            

        # Save results to Excel with original sheet names
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            prod_df.to_excel(writer, sheet_name="Production", index=False)
            Qreact_df.to_excel(writer,sheet_name="Reactive")
            price_df.to_excel(writer, sheet_name="LMP", index=False)
            flows_df.to_excel(writer, sheet_name="Flows", index=False)
            
        return "status: Optimal", True

    except Exception as e:
        print(f"Error in Bolognani model: {str(e)}")
        return f"Error: {str(e)}", False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_Bolognani_model.py <input_path> <output_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    status, success = run_decoupled_model(input_path, output_path)
    print(status)
    
    if not success:
        sys.exit(1)
