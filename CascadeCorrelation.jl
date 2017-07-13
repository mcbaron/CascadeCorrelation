# Adjusting weights and adding new hidden neurons
#
# Return:
# (w_io,w,w_0,w_hh,v,v_0)
# 1) w_io - weights from input to output units [output_neuron,input_neuron]
# 2) w - input-hidden weights [hidden_neuron,input_neuron]
# 3) w_0 - bias of input-hidden weights [hidden_neuron]
# 4) w_hh - hidden-hidden weights [hidden_neuron_to,hidden_neuron_from]
# 5) v - hidden-output weights [output_neuron,hidden_neuron]
# 6) v_0 - bias of hidden-output weights [output_neuron]

function cascade_correlation(n_input,training_set_in,training_set_out)

  # Parameters and variables
  alpha_hid = 0.1  # learning rate for new hidden unit's input weights
  n_candidates = 1  # how many candidate units will be initialized on adding each hidden neuron

  n_hidden = 0
  w_io = rand(1,n_input)  # weights input-output [output_neuron,input_neuron]
  v_0 = rand(1)  # bias of output neuron

  # Adjusting input-output weights by Delta Rule as much as possible
  (w_io, v_0) = delta(n_input,n_hidden,training_set_in,training_set_out,0,0,w_io,v_0,0,0)[1:2]

  # Adding first neuron into the Network
  # (Initializing several candidate units, training them, then choosing the best one)
  n_hidden = 1 # amount of hidden neurons (initially 1)

  w = zeros(1,n_input)  # weights (input-hidden) [hidden_neuron,input_neuron]
  w_0 = zeros(1)  # biases of each hidden neuron
  w_hh = zeros(1,1) # weights (hidden-hidden) [hidden_neuron_to,hidden_neuron_from]
  v = zeros(1,1) # weights (hidden-output) [output_neuron,hidden_neuron]

  # Candidate units
  w_cand = zeros(n_candidates,1,n_input)
  w_0_cand = zeros(n_candidates,1)
  w_hh_cand = zeros(n_candidates,1)  # can only receive outputs of other units

  z = zeros(n_hidden) # TODO calculated values at the outputs of each hidden neuron

  eps = 0.01  # precision
  err_prev = 0
  err = Inf

  # Calculating error and adding another hidden unit if needed
  for iteration=1:200 # (maximum 200 hidden units)

    # Incremental squared error (to decide if we need another hidden unit)
    err_prev = err
    err = 0

    # Best weights among candidate units
    w_best_cand = w_cand[1,:,:]
    w_0_best_cand = w_0_cand[1,:]
    w_hh_best_cand = w_hh[1,:,:]
    err_min = Inf # will definetly be less than Inf after Adjusting inputs of hidden unit

    # Making some candidate units with different initial weights,
    # then optimizing them as much as possible
    for c = 1:n_candidates

      w_cand[c,:,:] = rand(1,n_hidden,n_input)  # weights (input-hidden) [hidden_neuron,input_neuron]
      w_0_cand[c,:] = rand(1,n_hidden)  # biases of each hidden neuron
      w_hh_cand[c,:] = [rand(1,n_hidden-1) 0] # weights (hidden-hidden) [hidden_neuron_from]
      # TODO what if n_hidden=1 ?
      err_cand = 0  # correlation between output of hidden unit and residual output error of the network (to decide which candidate unit is best)

      if (iteration == 1) # if no hidden units yet
        (w_cand[c,:,:],w_0_cand[c,:],w_hh_cand[c,:],err_cand) =
        adjust_hidden(n_input,n_hidden,0,0,0,0,v_0,w_io,training_set_in,training_set_out,alpha_hid,w_cand[c,:,:],w_0_cand[c,:],w_hh_cand[c,:])
      else
        (w_cand[c,:,:],w_0_cand[c,:],w_hh_cand[c,:],err_cand) = adjust_hidden(n_input,n_hidden,w,w_0,v,w_hh,v_0,w_io,training_set_in,training_set_out,alpha_hid,w_cand[c,:,:],w_0_cand[c,:],w_hh_cand[c,:])
      end

      if (err_cand < err_min)  # if candidate is better
        w_best_cand = w_cand[c]
        w_0_best_cand = w_0_cand[c]
        w_hh_best_cand = w_hh_cand[c]
        err_min = err_cand
      end
    end

    # New lines/columns are already added
    w[n_hidden,:] = w_best_cand
    w_0[n_hidden] = w_0_best_cand
    w_hh[n_hidden,:] = w_hh_best_cand

    # Retraining in-out and hid-out weights
    (w_io, v_0, v) = delta(n_input,n_hidden,training_set_in,training_set_out,w,w_0,w_io,v_0,v,w_hh)

    # Calculating error by summating differences between FF results and training set
    for p = 1:size(training_set_in,1)
      y = feedforward(training_set_in[p,:],n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)[2]
      err += (training_set_out[p] - y)^2
    end

    # If error is low enough, stop adjust hidden neurons
    if (abs(err - err_prev) < eps)
      break
    end

    # Adding new neuron into the network
    n_hidden += 1

    # Candidate units
    w_cand = zeros(n_candidates,n_hidden,n_input)
    w_0_cand = zeros(n_candidates,n_hidden)
    w_hh_cand = zeros(n_candidates,n_hidden)

    # Adding new rows/columns (empty)
    w = [w; rand(1,n_input)]  # giving random weights (input-hidden)
    w_0 = [w_0; rand(1)]  # biases (input-hidden)
    w_hh = [w_hh zeros(n_hidden-1,1)]
    w_hh = [w_hh; rand(1,n_hidden-1) 0]
    #v = [v rand(1)] #

    z = zeros(n_hidden)
    #(z, y) = feedforward(x,n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)  # outputs of hidden units and output of the network

  end

  return (w_io,w,w_0,w_hh,v,v_0)

end
