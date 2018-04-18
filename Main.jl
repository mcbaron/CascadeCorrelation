# Cascade Correlation algorithm implementation

workspace()

# Hyperparameters
# Learning rate
#learning_rate = 0.001
filename_data = "data.csv"

sigmoid(x) = tanh.(x)
sigmoid_der(x) = sech.(x).^2

include("FeedForward.jl")
include("Delta.jl")
include("AdjustHidden.jl")
include("AddHidden.jl")
include("CascadeCorrelation.jl")
include("ShufflePatterns.jl")
include("PlotDecisionBoundary.jl")
include("ReadData.jl")
import CSV.read
using PyPlot
#using Plots
#pyplot()

const n_input = 2 # amount of input neurons
#n_output = 1 # amount of output neurons

# Opening file with data
(training_set_in, training_set_out) = read_data(filename_data)

# Applying CC algorithm
w_io = w = w_0 = w_hh = v = v_0 = err_arr = 0.0
w = Float64(0)
(w_io, w, w_0, w_hh, v, v_0, err_arr) =
@time cascade_correlation(training_set_in, training_set_out)

# Preparing mesh and plotting it
plot_decision_boundary()

#plot(collect(1:length(err_arr)), err_arr)
