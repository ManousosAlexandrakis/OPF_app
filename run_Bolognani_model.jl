
# Packages
using DataFrames, JuMP,Gurobi, GLPK
using LinearAlgebra
using XLSX


####################################################################                                 ####################################################################
####################################################################          Data Handling          ####################################################################
####################################################################                                 ####################################################################




# Loading Excel sheets into DataFrames
# Get the filename argument from command line
# Check arguments


if length(ARGS) < 2
    println("Usage: julia run_opf.jl input.xlsx output.csv")
    exit(1)
end

input_path = ARGS[1]
output_path = ARGS[2]

println("Reading input from: $input_path")
println("Saving output to: $output_path")

# Now load your Excel sheets dynamically:
gen_data = DataFrame(XLSX.readtable(input_path, "gen"))
Edges = DataFrame(XLSX.readtable(input_path, "edges"))
bus_data = DataFrame(XLSX.readtable(input_path, "bus"))
load_data = DataFrame(XLSX.readtable(input_path, "load"))
slack_data = DataFrame(XLSX.readtable(input_path, "ext_grid"))
Upward_data = DataFrame(XLSX.readtable(input_path, "Upward"))
Downward_data = DataFrame(XLSX.readtable(input_path, "Downward"))

# # Sets explanation:
# K = all generators except slack bus
# L = all buses except slack bus and K

# # Sbase of the system
Ssystem = 1

# # Create a dictionary mapping edges' idx to FlowMax
Flowmax_edge_dict = Dict{Int, Float64}()
for row in eachrow(Edges)
    Flowmax_edge_dict[row.idx] = row.FlowMax
end

# # Data for slack bus (voltage magnitude,voltage degree,bus number)
slack_v = slack_data[1, :vm_pu]  
slack_degree = slack_data[1,:va_degree]
slack_bus = slack_data[1,:bus]

buses = bus_data[:, :bus]
edges_index = Edges[:,:idx]

# # Create a dictionary to store the index of each bus
slack_index = findfirst(bus_data[:, :bus] .== slack_bus)
bus_id_to_index = Dict(bus_data[setdiff(1:end, slack_index), :bus] .=> 1:size(bus_data, 1)-1)
bus_id_to_index[slack_bus] = size(bus_data, 1)

# # Total number of buses and edges
n = size(bus_data,1)
n_edges = length(Edges.from_bus)
Lines = 1:n_edges

# # Create an array to store all the Nodes
Nodes = Int[]
for row in eachrow(bus_data)
    bus = row.bus
    push!(Nodes, bus)
end

# # Create an array to store all the K buses
K_buses = Int[]
for row in eachrow(gen_data)
    bus_id = row.bus
    if bus_id !=slack_bus && !(bus_id in K_buses)
        push!(K_buses, bus_id)
    end
end
n_K_buses = length(K_buses)

# # Create an array to store all the L buses
L_buses = filter(bus -> !(bus in K_buses) && bus != slack_bus, Nodes)
n_L_buses = length(L_buses)

# # Create an array to store all the K buses along with the slack bus
slack_K_buses = Int[]
for row in eachrow(gen_data)
    bus_id = row.bus
    if !(bus_id in slack_K_buses)
        push!(slack_K_buses, bus_id)
    end
end

# # Create an array to store all the K and L buses
K_L_buses = setdiff(Nodes, [slack_bus])
n_K_L_buses = length(K_L_buses)


# # Create a map for all the buses(bus names to indexes)
# Slack gets the last index
# K buses get the first indexes 
# L buses get the indexes after the K buses

# slack bus has the biggest index
slack_map = Dict{Int,Int}()
slack_map[slack_bus] = size(bus_data, 1)

# # Create a map for all the K buses
# K buses get the first indexes
K_bus_mapping = Dict(bus_id => index for (index, bus_id) in enumerate(K_buses))

# # Create a map for all the L buses
# L buses get the indexes after the K buses
L_bus_mapping = Dict(bus_id => index for (index, bus_id) in enumerate(L_buses))

# # Combine all mappings into a single comprehensive mapping
complete_mapping = Dict{Any, Int}()

# Add all entries from K_bus_mapping to complete_mapping
for (bus_id, index) in K_bus_mapping
    complete_mapping[bus_id] = index
end

# # Add all entries from L_bus_mapping to complete_mapping
for (bus_id, index) in L_bus_mapping
    complete_mapping[bus_id] = index + length(K_buses)
end

# # Add the slack bus entry to complete_mapping
complete_mapping[slack_bus] = size(bus_data, 1)


# # Create the Mapped Admittance Matrix (for a simp)
global Y = zeros(Complex, n, n)
global Z_w_slack = zeros(Complex, n, n)
for row in eachrow(Edges)
    From_Bus = complete_mapping[row.from_bus]
    To_Bus = complete_mapping[row.to_bus]
    x = row.X_pu
    r = row.R_pu
    local z = r + x .* im
    local y_admit = 1 ./ z[1]
    Y[From_Bus, To_Bus] = -y_admit
    Y[To_Bus, From_Bus] = -y_admit
    Y[From_Bus, From_Bus] = Y[From_Bus, From_Bus] +  y_admit
    Y[To_Bus, To_Bus] = Y[To_Bus, To_Bus] +  y_admit
    Z_w_slack[From_Bus, To_Bus] = -1 ./y_admit
    Z_w_slack[To_Bus, From_Bus] = -1 ./y_admit
    Z_w_slack[From_Bus, From_Bus] = Z_w_slack[From_Bus, From_Bus] +  1 ./y_admit
    Z_w_slack[To_Bus, To_Bus] = Z_w_slack[To_Bus, To_Bus] +  1 ./y_admit
end

# # Create the Admittance Matrix excluding the row that refers to the slack bus
sl = slack_map[slack_bus]
Y_without_slack =Y[ 1:end .!= sl, 1:end .!= sl]

# # Create the Impedance Matrix excluding the slack bus
Z = inv(Y_without_slack)

# # Create the R and X Matrices excluding the slack bus
global R_matrix = zeros(Float64, n_K_L_buses, n_K_L_buses)
global X_matrix = zeros(Float64, n_K_L_buses, n_K_L_buses)

R_matrix = real.(Z)
X_matrix = imag.(Z)

# # Create Matrices that are used on the paper for the linear approximation
global R_KK = zeros(Float64, n_K_buses, n_K_buses)
global X_KK = zeros(Float64, n_K_buses, n_K_buses)
global R_KL = zeros(Float64, n_K_buses, n_L_buses)
global X_KL = zeros(Float64, n_K_buses, n_L_buses)

global R_LK = zeros(Float64, n_L_buses, n_K_buses)
global X_LK = zeros(Float64, n_L_buses, n_K_buses)
global R_LL = zeros(Float64, n_L_buses, n_L_buses)
global X_LL = zeros(Float64, n_L_buses, n_L_buses)

global X_KK_inv = zeros(Float64, n_K_buses, n_K_buses)

# # R_KK and X_KK
for k in K_buses
    for m in K_buses
        R_KK[K_bus_mapping[k], K_bus_mapping[m]] = R_matrix[complete_mapping[k], complete_mapping[m]]
        X_KK[K_bus_mapping[k], K_bus_mapping[m]] = X_matrix[complete_mapping[k], complete_mapping[m]]
    end
end

# # R_KL and X_KL
for k in K_buses
    for m in L_buses
            R_KL[K_bus_mapping[k], L_bus_mapping[m]] = R_matrix[complete_mapping[k], complete_mapping[m]]
            X_KL[K_bus_mapping[k], L_bus_mapping[m]] = X_matrix[complete_mapping[k], complete_mapping[m]]
    end 
end

# # R_LK and X_LK
for k in L_buses
    for m in K_buses
        R_LK[L_bus_mapping[k], K_bus_mapping[m]] = R_matrix[complete_mapping[k], complete_mapping[m]]
        X_LK[L_bus_mapping[k], K_bus_mapping[m]] = X_matrix[complete_mapping[k], complete_mapping[m]]

    end 
end

# # R_LL and X_LL
for k in L_buses

    for m in L_buses
        R_LL[L_bus_mapping[k], L_bus_mapping[m]] = R_matrix[complete_mapping[k], complete_mapping[m]]
        X_LL[L_bus_mapping[k], L_bus_mapping[m]] = X_matrix[complete_mapping[k], complete_mapping[m]]

    end 

end

# # X_KK_inverted
X_KK_inv = inv(X_KK)


# # Calculate the total active power demand (P) and reactive power demand (Q) for each node
global total_p = 0.0
global total_q = 0.0

total_pgen_pload = Dict{Int,Float64}()
total_qgen_qload = Dict{Int,Float64}()
for row in eachrow(load_data)
    bus = row.bus
    Pload = row.p_mw
    Qload = row.q_mvar
    total_pgen_pload[bus] = get(total_pgen_pload, bus, 0.0) - (Pload/Ssystem)
    total_qgen_qload[bus] = get(total_qgen_qload, bus, 0.0) - (Qload/Ssystem)
end

# # Reactive power production limits for each generator bus
Qmin = Dict{Int, Float64}()
Qmax = Dict{Int, Float64}()
for row in eachrow(gen_data)
    bus = row.bus
    Pgen = row.p_mw
    #vm = row.vm_pu
    Qmin[row.bus] = row.QRmin/Ssystem
    Qmax[row.bus] = row.QRmax/Ssystem
    # total_pgen_pload[bus] = get(total_pgen_pload, bus, 0.0) + Pgen / Ssystem
end

# # Create a dictionary to store generator buses' Marginal Cost, Minimum and Maximum Active Power production limits
PU = Dict{Int,Float64}()
minQ = Dict{Int,Float64}()
maxQ = Dict{Int,Float64}()
for row in eachrow(Upward_data)
    bus = row.Bus
    u_price = row.PU
    u_minquantity = row.MinQ
    u_maxquantity = row.MaxQ
    PU[bus] = u_price
    minQ[bus] = u_minquantity/Ssystem
    maxQ[bus] = u_maxquantity/Ssystem
end

# # Create an array to store all the buses that can offer upward flexibility
Upward_set = Int[]
for row in eachrow(Upward_data)
    bus = row.Bus
    if maxQ[bus] !=0 
        push!(Upward_set, bus)
    end
end

# # Create dictionaries for the Resistance and Reactance of each edge
Rij = Dict{Int64, Float64}()
Xij = Dict{Int64, Float64}()
for row in eachrow(Edges)
    edge = row.idx
    From = row.from_bus
    To = row.to_bus
    R_ij = row.R_pu
    X_ij = row.X_pu

    Rij[edge] = R_ij
    Xij[edge] = X_ij
end
####################################################################                                 ####################################################################
#################################################################### Mathematical optimization model ####################################################################
####################################################################                                 ####################################################################

### Create a mathematical optimization model using the Gurobi Optimizer as the solver

# GUROBI_ENV = Gurobi.Env()
# model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
# set_optimizer_attribute(model, "MIPGap", 0.0) 
# set_silent(model)
model = Model(GLPK.Optimizer)
set_optimizer_attribute(model, "msg_lev", 1)  # Basic logging
set_optimizer_attribute(model, "tm_lim", 300000) 

### Variables

@variable(model, V[Nodes])                                  # Variable representing voltage magnitudes of each node
@variable(model, delta[Nodes])                              # Variable representing voltage angles of each node
@variable(model, Q[slack_K_buses])                          # Variable representing Reactive Power production by generator buses
@variable(model, production[Upward_set])                    # Variable representing Active Power production by generator buses
@variable(model, active_power_k[Nodes])                     # Variable representing Active Power injection for each node
@variable(model, reactive_power_k[Nodes])                   # Variable representing Reactive Power injection for each node
@variable(model, f[edges_index])                            # Variable representing Active Power flow on each edge
@variable(model, f_q[edges_index])                          # Variable representing Reactive Power flow on each edge

### Constraints for the optimization problem

# # Limits for the Reactive Power produced by the slack and the K buses
@constraint(model, [k in slack_K_buses], Qmin[k] <= Q[k] <= Qmax[k]) 

# # Define the voltage magnitude and angle of the slack bus
@constraint(model, V[slack_bus] == slack_v)                           
@constraint(model, delta[slack_bus] == slack_degree) 

# # Limits for the Active Power produced by the slack and the K buses
@constraint(model, UpperBound1[i in Upward_set], production[i] <= maxQ[i])
@constraint(model, UpperBound2[i in Upward_set], production[i] >= minQ[i])

# # Voltage magnitude limits for all buses except the slack bus
@constraint(model,LowerBoundV[k in K_L_buses] ,0.8 <= V[k] ) 
@constraint(model,UpperBoundV[k in K_L_buses] , V[k] <= 1.2) 


# # Power Balance for each node
@constraint(model,injectionsandflows[i in Nodes], sum(f[j] for j in edges_index if Edges.from_bus[j] == i) - sum(f[j] for j in edges_index if Edges.to_bus[j] == i) == active_power_k[i])
@constraint(model,[i in Nodes], sum(f_q[j] for j in edges_index if Edges.from_bus[j] == i) - sum(f_q[j] for j in edges_index if Edges.to_bus[j] == i) == reactive_power_k[i])

# # Active Power Flows on each edge (Taylor Series Approximation)
@constraint(model, TaylorActiveFlow[i in edges_index],f[i] - (Rij[i] * (V[Edges.from_bus[i]] - V[Edges.to_bus[i]]) + Xij[i] * (delta[Edges.from_bus[i]] - delta[Edges.to_bus[i]])) 
/ (Rij[i]^2 + Xij[i]^2)  == 0 )

# # Reactive Power Flows on each edge (Taylor Series Approximation)
@constraint(model, TaylorReActiveFlow[i in edges_index], f_q[i] - (Xij[i] * (V[Edges.from_bus[i]] - V[Edges.to_bus[i]]) - Rij[i] * (delta[Edges.from_bus[i]] - delta[Edges.to_bus[i]])) 
/ (Rij[i]^2 + Xij[i]^2)    == 0)

# # Active Power Flow Limits
@constraint(model, FlowmaxUpper[i in edges_index],f[i] <=Flowmax_edge_dict[i])
@constraint(model, FlowmaxDown[i in edges_index],-f[i] <=Flowmax_edge_dict[i])

# # Reactive Power Flow Limits
@constraint(model, [i in edges_index],f_q[i] <=Flowmax_edge_dict[i])
@constraint(model, [i in edges_index],-f_q[i] <=Flowmax_edge_dict[i])

# # Voltage magnitude equation for all buses except the slack bus
@constraint(model, Voltage[k in K_L_buses], V[k] == slack_v + (sum(R_matrix[complete_mapping[k], complete_mapping[i]] * (active_power_k[i]) for i in K_buses)
+ sum(X_matrix[complete_mapping[k], complete_mapping[i]] * (reactive_power_k[i]) for i in K_buses)
+ sum(R_matrix[complete_mapping[k], complete_mapping[i]] * (active_power_k[i]) for i in L_buses)
+ sum(X_matrix[complete_mapping[k], complete_mapping[i]] * (reactive_power_k[i]) for i in L_buses)) / slack_v)

# # Voltage angle equation for all buses except the slack bus
@constraint(model, DeltaConstraint[k in K_L_buses], delta[k] == slack_degree + (sum(-R_matrix[complete_mapping[k], complete_mapping[i]] * (reactive_power_k[i]) for i in K_buses)
+ sum(X_matrix[complete_mapping[k], complete_mapping[i]] * (active_power_k[i]) for i in K_buses)
+ sum(-R_matrix[complete_mapping[k], complete_mapping[i]] * (reactive_power_k[i]) for i in L_buses)
+ sum(X_matrix[complete_mapping[k], complete_mapping[i]] * (active_power_k[i]) for i in L_buses)) / (slack_v^2))

# # Fixing Voltage Magnitude for generator-buses
@constraint(model, fixed[i in Upward_set], V[i]==1)


# # Reactive and Active Power injection equations for all buses
@constraint(model,reactive_power[k in Nodes],reactive_power_k[k] == sum(Q[i] for i in slack_K_buses if i == k) + get(total_qgen_qload,k,0))
@constraint(model,  active_power[k in Nodes],  active_power_k[k] == get(total_pgen_pload,k,0) + sum(production[i] for i in Upward_set if i == k) ) #dual-variable is the nodal price

# # Objective Function
@objective(model, Max,  -  sum(PU[i]*production[i] for i in Upward_set))

# # Solve the optimization problem
optimize!(model)

# # Dual Variables for pricing

# for k in Nodes
#      println("ρ$k = ",dual(active_power[k]))
# end

# # Dual Variables

# for k in Nodes
#     println("σ$k = ",dual(injectionsandflows[k]))
# end
# for k in Upward_set
#     println(dual(fixed[k]))
# end
# for k in edges_index
#     println("ψ$k = ",dual(TaylorActiveFlow[k]))
# end
# for k in edges_index
#     println("φ$k = ",dual(TaylorReActiveFlow[k]))
# end
# for k in edges_index
#     println("λ+$k = ",dual(FlowmaxUpper[k]))
# end
# for k in edges_index
#     println("λ-$k = ",dual(FlowmaxDown[k]))
# end

# for k in K_L_buses
#     println("ν$k = ",dual(Voltage[k]))
# end
# for k in K_L_buses
#     println("η$k = ",dual(DeltaConstraint[k]))
# end
# for k in K_L_buses
#     println("ξ+$k = ",dual(UpperBoundV[k]))
# end
# for k in K_L_buses
#     println("ξ-$k = ",dual(LowerBoundV[k]))
# end
# for k in Upward_set
#     println("μ+$k = ",dual(UpperBound1[k]))
# end
# for k in Upward_set
#     println("μ-$k = ",dual(UpperBound2[k]))
# end

####################################################################                                      ####################################################################
#################################################################### Results for the optimization problem ####################################################################
####################################################################                                      ####################################################################

# # Results for the Voltage Magnitude and Voltage Angle
results_df = DataFrame(
    Bus = Nodes,
    vm_pu = [value(V[i]) for i in Nodes],
    va_degree = [rad2deg(value(delta[i])) for i in Nodes]
)

# # Results for the Active Power production
prod_df = DataFrame(   
    bus = Upward_set,
    production = [value(production[i]) for i in Upward_set],
    pmax = [maxQ[i] for i in Upward_set],
    pmin =[minQ[i] for i in Upward_set],
    PU = [PU[i] for i in Upward_set],
)

# # Results for the Reactive Power production
Qreact_df = DataFrame(
    Bus = Upward_set,
    q_pu = [round(value(Q[i]), digits=3) for i in Upward_set],
    qmin = [Qmin[i] for i in Upward_set],
    qmax =[Qmax[i] for i in Upward_set]
)

# # Results for Power Injections
PowerInjection_df = DataFrame(
     Bus = Nodes,
     q_injection = [round(value(reactive_power_k[i]), digits=3) for i in Nodes],
     p_injection = [round(value(active_power_k[i]), digits=3) for i in Nodes],

 )

# Results for Prices
price_df = DataFrame(
    Bus = buses,
    price = [round(-dual(active_power[j]), digits=3) for j in buses]
)

# # Results for Flows     
Edges_leng = 1:length(Edges.from_bus)
from_bus = Edges[:,:from_bus]
to_bus = Edges[:,:to_bus]
flows_df = DataFrame(
    Edge = edges_index,
     from_bus = [from_bus[i] for i in Edges_leng],
    flows_to = [to_bus[i] for i in Edges_leng],
    Flow_Active = [value(f[i]) for i in Edges_leng],
    Flow_Reactive = [value(f_q[i]) for i in Edges_leng],
    Flowmax = [Flowmax_edge_dict[i] for i in Edges_leng ]
)

# # Print the results
# println("")
# println("Voltage magnitudes [p.u.] and Voltage angles [°]:")
# println(results_df)
# println("")
# println("Active power production [p.u.]:")
# println(prod_df)
# println("")
# println("Reactive power production [p.u.]:")
# println(Qreact_df)
# println("")
# println("Nodal prices [€/MWh]:")
# println(price_df)
# println("")
# println("Active and Reactive power flows for lines [p.u.]:")
# println(flows_df)
# println("")
# println("Active and reactive power injections [p.u.]:")
# println(PowerInjection_df)
# println("")
println("Termination Status:", termination_status(model))


# # Results stored in an XLSX file
# filepath2 = "/Users/malexandrakis/Documents/Results/Paper_nodes_PV/LINEAR_OPF_ehv1.xlsx"
#XLSX.writetable("output/Bolognani_opf_results.xlsx",  "Results" => results_df , "Production" => prod_df ,  "Reactive_Production" => Qreact_df, "Price" => price_df,"Flows"=> flows_df)   
output_dir = "output"
if !isdir(output_dir)
    mkdir(output_dir)
end

# Results stored in an XLSX file in the output folder
output_file = joinpath(output_dir, basename(output_path))
XLSX.writetable(output_file,
    "Results" => results_df,
    "Production" => prod_df,
    "Reactive_Production" => Qreact_df,
    "Price" => price_df,
    "Flows" => flows_df,
    "Power_Injection" => PowerInjection_df
)

println("Results saved to: ", output_file)
println("Termination Status:", termination_status(model))


