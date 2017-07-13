# Adding hidden neuron
# Applying gradient ascent on the input weights of the candidate hidden unit
# n_hidden - number of expected hidden units; one of them not connected yet

function adjust_hidden(n_input,n_hidden,w,w_0,v,w_hh,v_0,w_io,training_set_in,training_set_out,alpha_hid,w_cand_concr,w_0_cand_concr,w_hh_cand_concr)

  eps = 0.1 # patience for adjusting input weights of the candidate
  err_prev = 0
  err = Inf # incremental error for the candidate hidden unit

  for iter=1:200 # iterations for gradient ascent; endless loop protection

    err_prev = err
    err = 0

    # Shuffle patterns
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # Calculating error after adding this hidden unit
    z_avg = 0 # average output of the new hidden neuron (not yet connected) among the patterns
    e_avg = 0 # average output error among patterns
    z_pattern = zeros(size(training_set_in,1))  # hidden unit output values after feed-forward
    y_pattern = zeros(size(training_set_in,1))  # output unit values after feed-forward
    summ = zeros(size(training_set_in,1)) # calculated weighted sums of inputs' of last added hidden neuron

    # Calculating averages for hidden unit values and output error
    for i=1:1 # for each hidden unit (?) TODO
      for j=1:size(training_set_in,1) # for each training pattern
        y_pattern[j] = feedforward(training_set_in[j,:],n_input,w,w_0,n_hidden-1,v,v_0,w_hh,w_io)[2]  # network output for each pattern
        e_avg += (training_set_out[j] - y_pattern[j])
        z_pattern[j] = w_cand_concr' * training_set_in[j,:]  # output of new hidden unit for pattern
        z_avg += z_pattern[j]
      end
      z_avg = z_avg / size(training_set_in, 1)  # average output value of the hidden unit
      e_avg = e_avg / size(training_set_in, 1)  # average residual error

      for j=1:size(training_set_in,1) # for each training pattern
        err += abs((z_pattern[j]-z_avg)*(y_pattern[j]-e_avg))
      end
    end

    # Applying gradient ascent (optimizing input weights of the new hidden neuron (NHN))

    # Input-hidden weights bias of NHN
    d_w_0_cand_concr = 0
    for i=1:size(training_set_in,1)
      d_w_0_cand_concr += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * 1
    end
    w_0_cand_concr += d_w_0_cand_concr  # changing w_0 of the candidate unit

    # Input-hidden weights of NHN
    for j=1:n_input
      d_w_0_cand_concr = 0
      for i=1:size(training_set_in,1)
        d_w_0_cand_concr += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * training_set_in[i,j]
      end
      w_cand_concr[j] += d_w_0_cand_concr
    end

    # Hidden-hidden weights of NHN
    for j=1:n_hidden-1
      d_b_cand_concr = 0
      for i=1:size(training_set_in,1)
        d_b_cand_concr += ((training_set_out[i] - y_pattern[i]) - e_avg) * sigmoid_der(summ[i]) * z_pattern[i]
      end
      w_hh_cand_concr[j] += d_b_cand_concr
    end

    if (abs(err - err_prev) < eps)  # if error not improving, stop
      break
    end

  end

  return (w_cand_concr,w_0_cand_concr,w_hh_cand_concr,err)

end
