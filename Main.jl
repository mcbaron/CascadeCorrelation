#!/usr/bin/env julia

# Cascade Correlation algorithm implementation

workspace()

include("FeedForward.jl")
include("Delta.jl")
include("CascadeCorrelation.jl")
include("ShufflePatterns.jl")

n_input = 2 # amount of input neurons
#n_output = 1 # amount of output neurons
sigmoid(x) = tanh(x)
sigmoid_der(x) = sech(x)^2

training_set_in = [5 0; 4 0.4; 5 5; 1 5; 1 1; 1 5; 0 2; 4 4; 3 6; 4 0.2;
                   0.5 0.2; 4.5 0.2; 3 3; 3 1; 4 3; 3.5 0.1; 4 0.1; 1.5 1; 0.5 1.2; 2 3;
                   1 0.5; 4 0.1; 0 1; 6 0; 6 2; 5 0.5; 4 0.1]
training_set_out = [1; 1; -1; -1; 1; -1; 1; -1; -1; 1;
                    1; 1; -1; -1; -1; 1; 1; 1; 1; -1;
                    1; 1; 1; -1; -1; -1; 1]

#training_set_in =   [0 1; 0 3; 0 5; 1 1; 1 2; 1 3; 1 4; 1 5; 1 6;
#3 2; 5 2; 2 1; 2 0; 3 6; 4 3; 4 6; 6 5; 6 1; 5 3; 5 4; 2 6; 4 6;
#7 3; 7 6; 4 4.5; 2 5; 5.5 3; 7 7]
#training_set_out =  permutedims([-1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 1 1 -1 -1 -1 -1 1 -1 1 -1], [2,1])

using PyPlot
x_plot = linspace(0.1,7,70)
y_plot = linspace(0.1,7,70)
z_plot = zeros(70,70)

# Applying Cascade Correlation algorithm
(w_io,w,w_0,w_hh,v,v_0) = cascade_correlation(n_input,training_set_in,training_set_out)

# Testing and plotting
for i=1:1:70
  for j=1:1:70
    z_plot[i,j] = feedforward([x_plot[i] y_plot[j]],n_input,w,w_0,size(w,1),v,v_0,w_hh,w_io)[2][1]
  end
end

surf(x_plot, y_plot, z_plot)
plot3D(training_set_in[:,1], training_set_in[:,2], training_set_out, marker='.')
