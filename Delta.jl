# Delta Rule implementation

function delta(n_input,n_hidden,training_set_in,training_set_out,w,w_0,w_io,v_0,v,w_hh)

  alpha = 0.1 # learning rate
  eps = 0.1 # patience
  err = Inf # squared error between y and y_target
  err_prev = 0 # same error on previous iteration
  n_patterns = size(training_set_out,1)

  for iter=1:500 # up to maximum amount of iterations (endless loop protection)

    err_prev = err
    err = 0
    sum_y = 0 # weighted sum of inputs of the output neuron
    alpha = alpha * 0.99  # decreasing learning rate

    d_v_0 = zeros(size(v_0))
    d_v = zeros(size(v))
    d_w_io = zeros(size(w_io))

    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)  # shuffle patterns

    for i=1:n_patterns  # for each training pattern
      (z, y) = feedforward(training_set_in[i,:],n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)[1:2]
      sum_y = feedforward(training_set_in[i,:],n_input,w,w_0,n_hidden,v,v_0,w_hh,w_io)[4]
      y = y[1]

      # Applying gradient descent
      for j=1:1 # for each output unit
        d_v_0[j] += -alpha * (training_set_out[i]-y) * sigmoid_der(sum_y) * 1
        for k=1:n_input # for each input unit
          d_w_io[j,k] += -alpha * (training_set_out[i]-y) * sigmoid_der(sum_y) * training_set_in[i,k]
        end
        for k=1:n_hidden
          d_v[k] += -alpha * (training_set_out[i]-y) * sigmoid_der(sum_y) * z[k]
        end
      end

      # Increment error between target and calculated output
      err += (training_set_out[i] - y)^2
    end

    # Batch
    v_0 += d_v_0
    w_io += d_w_io
    v += d_v

    if (abs(err - err_prev) < eps)
      break
    end
  end

  return (w_io, v_0, v)

end
