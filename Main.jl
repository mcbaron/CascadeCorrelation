#!/usr/bin/env julia
# Cascade Correlation algorithm implementation

workspace()

include("FeedForward.jl")
include("Delta.jl")
include("CascadeCorrelation.jl")
include("ShufflePatterns.jl")
include("AdjustHidden.jl")
include("PlotDecisionBoundary.jl")

n_input = 2 # amount of input neurons
#n_output = 1 # amount of output neurons
sigmoid(x) = tanh(x)
sigmoid_der(x) = sech(x)^2

# Opening file with data
input_file = "data.csv"
using DataFrames
data_frame = readtable(input_file, separator = ',')
training_set_in = convert(Matrix, data_frame[:,1:2])
training_set_out = convert(Array, data_frame[:,3])

# Applying CC algorithm
w_io = w = w_0 = w_hh = v = v_0 = 0
(w_io, w, w_0, w_hh, v, v_0) = cascade_correlation(training_set_in, training_set_out)

# Preparing mesh and plotting it
using PyPlot
plot_decision_boundary()
