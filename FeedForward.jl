# Feed-Forward of the CC Neural Network
# x - input units
# n_input - amount of input values
# w - weights (input-hidden)
# w_0 - biases of hidden neurons
# n_hidden - amount of hidden neurons
# v - weights (hidden-output)
# v_0 - bias of the output neuron
# b - weights (hidden-hidden)
# c - weights (input-output)

function feedforward(x, n_input, w, w_0, n_hidden, v, v_0, w_hh, w_io)

  z = zeros(n_hidden) # calculated values of each hidden neuron (sigmoid applied)
  y = 0 # output
  sum_h::Float64 = 0 # weighted sum of inputs of the last added hidden neuron
  sum_y::Float64 = 0 # weighted sum of inputs of the output neuron

  for i=1:n_hidden # for each hidden neuron
    sum_h = 0
    sum_h += w_0[i]  # bias (x_0 = 1)
    for j=1:n_input # input-hidden
      sum_h += w[i,j]*x[j]
    end
    for j=1:i-1 # for each preceding hidden neuron
      sum_h += w_hh[i,j]*z[j]
    end
    z[i] = tanh.(sum_h)[1]  # output of hidden neuron
  end

  for i=1:1 # for each output neuron
    sum_y = 0
    sum_y += v_0  # bias (z_0 = 1)
    for j=1:n_input # for each input neuron
      sum_y += w_io[i,j]*x[j]
    end
    for j=1:n_hidden
      sum_y += (v[i]*z[j])
    end
    y = tanh.(sum_y)[1] # TODO delete
  end

  return (z, y[1], sum_h, sum_y[1]) # returns tuple:
  # z - output values of hidden units
  # y - output value of network
  # sum_h - weighted sum of inputs of last added hidden unit
  # sum_y - weighted sum of inputs of output unit
end
