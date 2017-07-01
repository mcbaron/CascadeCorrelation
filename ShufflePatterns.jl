# Shuffle training patterns in a random order

function shuffle_patterns(training_set_in, training_set_out)

  n_patterns = size(training_set_in, 1)
  temp_arr = zeros(n_patterns)
  temp_bool = zeros(n_patterns)
  ret_in = zeros(size(training_set_in))
  ret_out = zeros(size(training_set_out))
  for i=1:n_patterns
    temp_arr[i] = i
  end
  temp_arr = shuffle(temp_arr)
  temp_arr = convert(Array{Int,1}, temp_arr)
  for i=1:n_patterns
    ret_in[i,:] = training_set_in[temp_arr[i],:]
    ret_out[i] = training_set_out[temp_arr[i]]
  end

  return (ret_in, ret_out)

end
