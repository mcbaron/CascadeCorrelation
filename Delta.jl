# Delta Rule implementation

function delta( nn_model,
                training_set_in,
                training_set_out,
                learning_rate_out,
                eps_delta,
                max_iter_delta)

  # Squared error between y and y_target
  err = Inf
  err_prev = 0.0

  # Amount of examples
  n_examples = size(training_set_out, 1)

  # Gradient descent iterations (with endless loop protection)
  for iter=1:max_iter_delta

    err_prev = err
    err = 0.0
    # Weighted sum of inputs of the output unit
    sum_y = 0.0
    # Decrease learning rate
    learning_rate_out *= 0.98

    # Batch gradient descent (ascent)
    d_v_0 = zeros(size(nn_model.v_0))
    d_v = zeros(size(nn_model.v))
    d_w_io = zeros(size(nn_model.w_io))

    # Shuffle examples
    (training_set_in, training_set_out) = shuffle_patterns(training_set_in, training_set_out)

    # Iterate through each example
    for i=1:n_examples
      (z, y) = feedforward(training_set_in[i,:], nn_model)[1:2]
      sum_y = feedforward(training_set_in[i,:], nn_model)[4]
      y = y[1]

      # Difference between target output value and calculated one
      #e_out = training_set_out[i] - y

      # ----- GRADIENT DESCENT -----
      # Through output units
      for j=1:1
        d_v_0 += learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) * 1
        # Through input-output weights
        for k=1:nn_model.n_input
          d_w_io[j,k] += learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) * training_set_in[i,k]
        end
        # Through hidden-output weights
        for k=1:nn_model.n_hidden
          d_v[k] += learning_rate_out * (training_set_out[i] - y) * activation_der(sum_y) * z[k]
        end
      end

      # Increment error between target and calculated output
      err += (training_set_out[i] - y)^2
    end

    # Batch update
    nn_model.v_0 += d_v_0
    nn_model.w_io += d_w_io
    nn_model.v += d_v
    nn_model.v_0 = nn_model.v_0[1]

    # Check precision and break if needed
    if (abs(err - err_prev) < eps_delta)
      break
    end
  end

  return (nn_model::NN_model, err)

end
