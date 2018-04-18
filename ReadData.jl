
function read_data(input_file)

  #data_frame = readtable(input_file, separator = ',')
  data_frame = CSV.read(input_file)
  training_set_in::Array{Float64,2} = convert(Matrix, data_frame[:,1:2])
  training_set_out::Array{Float64,1} = convert(Array, data_frame[:,3])

  return (training_set_in, training_set_out)

end
