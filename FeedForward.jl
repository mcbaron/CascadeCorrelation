# Feed-Forward of the CC Neural Network
# x - input units [n_examples, n_dims]
# nn_model - CCNN model

function feedforward(x::Array{Float64,1}, nn_model)

  z = zeros(nn_model.n_hidden)  # calculated outputs of each hidden unit (activation function applied)
  y = 0.0                       # output of the network
  sum_h::Float64 = 0.0          # weighted sum of inputs of the last added hidden unit
  sum_y::Float64 = 0.0          # weighted sum of inputs of the output unit

  # Iterate through hidden units
  for i=1:nn_model.n_hidden
    # Weighted sum of inputs for the hidden unit
    sum_h = 0.0
    sum_h += nn_model.w_0[i]  # bias (x_0 = 1)
    # Input-hidden connections
    for j=1:n_input
      sum_h += nn_model.w[i,j] * x[j]
    end
    # Iterate through preceding hidden units; hidden-hidden connections
    for j=1:i-1
      sum_h += nn_model.w_hh[i,j] * z[j]
    end
    # Output of the hidden unit with activation applied
    z[i] = activation(sum_h)[1]
  end

  # Iterate through output units (one output)
  for i=1:1
    # Weighted sum of inputs for the output unit
    sum_y = 0.0
    sum_y += nn_model.v_0  # bias (z_0 = 1)
    # Input-output connections
    for j=1:n_input
      sum_y += nn_model.w_io[i,j] * x[j]
    end
    # Hidden-output connections
    for j=1:nn_model.n_hidden
      sum_y += (nn_model.v[i] * z[j])
    end
    # Output of the network with activation applied
    y = activation(sum_y)[1]
  end

  return (z, y[1], sum_h, sum_y[1]) # returns tuple:
  # z - output values of hidden units
  # y - output value of network
  # sum_h - weighted sum of inputs of last added hidden unit
  # sum_y - weighted sum of inputs of output unit
end
