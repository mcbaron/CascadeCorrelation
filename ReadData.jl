
function read_data(input_file)

  data_frame = readtable(input_file, separator = ',')
  training_set_in = convert(Matrix, data_frame[:,1:2])
  training_set_out = convert(Array, data_frame[:,3])

  return (training_set_in, training_set_out)

end
