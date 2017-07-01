# Adjusting weights and adding new hidden neurons

function cascade_correlation(n_input,training_set_in,training_set_out)

  # Parameters and variables
  alpha_hid = 0.1  # learning rate for new hidden unit's input weights
  n_hidden = 0
  w_io = rand(1,n_input)  # weights input-output [output_neuron,input_neuron]
  v_0 = rand(1)  # bias of output neuron

  # Adjusting input-output weights by Delta Rule as much as possible
  (w_io, v_0) = delta(n_input,n_hidden,training_set_in,training_set_out,0,0,w_io,v_0,0,0)[1:2]

  # Adding first neuron into the Network
  n_hidden = 1 # amount of hidden neurons (initially 1)
  w = rand(1,n_input)  # weights (input-hidden) [hidden_neuron,input_neuron]
  w_0 = rand(1)  # biases of each hidden neuron
  v = rand(1,1) # weights (hidden-output) [output_neuron,hidden_neuron]
  w_hh = zeros(1,1) # weights (hidden-hidden) [hidden_neuron_to,hidden_neuron_from]

  z = zeros(n_hidden) # calculated values at the outputs of each hidden neuron

  eps = 0.01
  err = Inf
  err_prev = 0
  # Calculating error and adding another hidden unit if needed
  for iteration=1:200

    err_prev = err
    err = 0 # incremental error for the candidate hidden unit
    alpha_hid = alpha_hid * 0.99  # decrease learning rate

    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)  # shuffle patterns

    # Calculating error after adding hidden unit
    z_avg = 0
    e_avg = 0
    z_pattern = zeros(size(training_set_in,1),n_hidden)  # results of hidden unit values after feed-forward
    y_pattern = zeros(size(training_set_in,1))  # results of output unit value after feed-forward
    summ = zeros(size(training_set_in,1)) # calculated weighted sums of input of last added hidden neuron

    # Calculating averages for hidden unit values and output error
    for i=1:1 # for each output unit
      for j=1:size(training_set_in,1) # for each training pattern
        (z_pattern[j,:], y_pattern[j,:], summ[j]) = feedforward(training_set_in[j,:],n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)[1:3]
        z_avg += z_pattern[j]
        e_avg += (training_set_out[j] - y_pattern[j])
      end
      z_avg = z_avg / size(training_set_in, 1)
      e_avg = e_avg / size(training_set_in, 1)

      for j=1:size(training_set_in,1) # for each training pattern
        err += abs((z_pattern[j][1]-z_avg)*(y_pattern[j]-e_avg))
      end
    end

    # Applying gradient ascent (optimizing input weights of the new hidden neuron (NHN))

    # Input-hidden weights bias of NHN
    d_w_0 = 0
    for i=1:size(training_set_in,1)
      d_w_0 += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * 1
    end
    w_0[n_hidden] += d_w_0

    # Input-hidden weights of NHN
    for j=1:n_input
      d_w = 0
      for i=1:size(training_set_in,1)
        d_w += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * training_set_in[i,j]
      end
      w[n_hidden,j] += d_w
    end

    # Hidden-hidden weights of NHN
    for j=1:n_hidden-1
      d_b = 0
      for i=1:size(training_set_in,1)
        d_b += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * z_pattern[i,j]
      end
      w_hh[n_hidden,j] += d_b
    end

    # Retraining in-out and hid-out weights
    (w_io, v_0, v) = delta(n_input,n_hidden,training_set_in,training_set_out,w,w_0,w_io,v_0,v,w_hh)

    # If error is low enough, stop adding hidden neurons
    if (abs(err - err_prev) < eps)
      break
    end

    # Adding new neuron into the network
    n_hidden += 1
    w = [w; rand(1,n_input)]  # giving random weights (input-hidden)
    w_0 = [w_0; rand(1)]  # biases (input-hidden)
    v = [v rand(1)] #
    w_hh = [w_hh zeros(n_hidden-1,1)]
    w_hh = [w_hh; rand(1,n_hidden-1) 0]

    z = zeros(n_hidden)
    #(z, y) = feedforward(x,n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)  # outputs of hidden units and output of the network

  end

  return (w_io,w,w_0,w_hh,v,v_0)

end
