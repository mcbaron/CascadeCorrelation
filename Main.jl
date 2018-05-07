# Cascade Correlation

using CSV
using PyPlot

# Hyperparameters
const learning_rate_hid_in = 0.01 # learning rate for input-hidden weights (when adding hidden unit)
const learning_rate_out = 0.01 # learning rate for hidden-output and input-output weights (for delta rule retraining)
const eps_delta = 0.001 # precision for (re)training input-output and hidden-ouput weights
const eps_cascade = 0.01  # precision for adding hidden units (if after adding hidden unit error decrease is less then eps_cascade, stop adding)
const max_iter_delta = 1000 # max iterations for -output retraining
const max_iter_cand = 500 # max iterations for candidate unit training (input-hidden_candidate)
const n_candidates = 20 # how many candidate units will be initialized on adding each hidden neuron
const max_hidden = 2 # maximum amount of hidden units

# File with CSV data
filename_data = "data/data.csv"

# Activation functions for all units (hidden and output)
activation(x) = tanh.(x)
activation_der(x) = sech.(x).^2

include("CascadeCorrelation.jl")
include("PlotDecisionBoundary.jl")
include("ReadData.jl")

# Define a network model
mutable struct NN_model
  n_input     # input units
  n_hidden    # hidden units
  w_io    # input-output weights
  w       # input-hidden weights
  w_0     # input-hidden bias
  w_hh    # hidden-hidden weights
  v       # hidden-output weights
  v_0     # hidden-output bias
end

# Read data from file
(training_set_in, training_set_out) = read_data(filename_data)
# Amount of input units
n_input = size(training_set_in, 2)

# Train the model using CC
nn_model, err_arr =
  @time cascade_correlation(training_set_in, training_set_out, learning_rate_hid_in, learning_rate_out, eps_delta, max_iter_delta)

# Preparing mesh and plotting it
plot_decision_boundary(nn_model, err_arr)
