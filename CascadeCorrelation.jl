" Adjusting weights and adding new hidden neurons
# Arguments:

- `training_set_in`: [pattern, input]
- `training_set_out`: [pattern] (always 1 output)

# Return:

- `w_io` - weights from input to output units [output_neuron,input_neuron]
- `w` - input-hidden weights [hidden_neuron,input_neuron]
- `w_0` - bias of input-hidden weights [hidden_neuron]
- `w_hh` - hidden-hidden weights [hidden_neuron_to,hidden_neuron_from]
- `v` - hidden-output weights [output_neuron,hidden_neuron]
- `v_0` - bias of hidden-output weights [output_neuron]
"

function cascade_correlation(training_set_in::Array{Float64,2}, training_set_out::Array{Float64,1})

  # Parameters and variables
  n_input = size(training_set_in,2)
  const alpha_hid_in = 0.01  # learning rate for new hidden unit's input weights
  const n_candidates = 10  # how many candidate units will be initialized on adding each hidden neuron
  const max_hidden = 10  # maximum amount of hidden units

  # Initialization
  n_hidden = 0
  w_io = rand(1,n_input)  # weights input-output [output_neuron,input_neuron]
  v_0 = rand()  # bias of output neuron

  # Adjusting input-output weights by Delta Rule as much as possible
  (w_io, v_0) = delta(n_input, n_hidden, training_set_in, training_set_out, zeros(0,0), zeros(0,0), w_io, v_0, zeros(0,0), zeros(0,0))[1:2]

  # Adding first neuron into the Network (Initializing several candidate units, training them, then choosing the best one)
  n_hidden = 0

  w = zeros(0,n_input)  # weights (input-hidden) [hidden_neuron,input_neuron]
  w_0 = zeros(0)  # biases of each hidden neuron
  w_hh = 0.0 # weights (hidden-hidden) [hidden_neuron_to,hidden_neuron_from]
  v = rand(1,1) # weights (hidden-output) [output_neuron,hidden_neuron]

  z = zeros(n_hidden) # TODO calculated values at the outputs of each hidden neuron

  const eps = 0.01  # precision (if after adding hidden unit error decreases less then eps, stop adding)
  err_prev = 0.0
  err = Inf
  err_arr = zeros(max_hidden) # error of prediction for every amount of hidden units

  # Calculating prediction error and adding another hidden unit if needed
  for iteration = 1:max_hidden

    # Incremental squared error (to decide if we need another hidden unit)
    err_prev = err
    err = 0.0

    (w, w_0, w_hh, v, n_hidden) = add_hidden(training_set_in, training_set_out, w, w_0, w_hh, v, v_0, w_io, n_candidates, n_hidden, alpha_hid_in)[1:5]

    # Retraining in-out and hid-out weights
    (w_io, v_0, v, err) = delta(n_input,n_hidden,training_set_in,training_set_out,w,w_0,w_io,v_0,v,w_hh)

    # Calculating error by summating differences between FF results and training set
    for p = 1:size(training_set_in,1)
      y = feedforward(training_set_in[p,:],n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)[2]
      err += (training_set_out[p] - y)^2
    end

    # History of errors for each amount of hidden units
    err_arr[iteration] = err

    # If error is low enough, stop adding hidden units
    if (abs(err - err_prev) < eps)
      break
    end

  end

  print("\nCascade Correlation training completed\n")
  print("Hidden neurons:",n_hidden,"\n\n")

  return (w_io, w, w_0, w_hh, v, v_0, err_arr)

end
