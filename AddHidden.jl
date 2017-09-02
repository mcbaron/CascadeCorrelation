
function add_hidden(training_set_in, training_set_out, w, w_0, w_hh, v, v_0, w_io, n_candidates, n_hidden, alpha_hid_in)

	n_input = size(training_set_in,2)
	n_out = size(training_set_out,2)

	# Candidate units
	w_cand = Array{Array{Float64}}(n_candidates)  # input -> new_hidden
	w_0_cand = zeros(n_candidates)	# new hidden bias
	w_hh_cand = Array{Array{Float64}}(n_candidates)  # hidden -> hidden; can only receive outputs of other units

	# Weights of best candidate unit so far
	w_best_cand = 0
	w_0_best_cand = 0
	w_hh_best_cand = 0
	err_min = Inf # minimal error among candidates, will definetly be less than Inf after Adjusting inputs of hidden unit

	# Making some candidate units with different initial weights,
	# then optimizing them as much as possible
	for c = 1:n_candidates

		w_cand[c] = rand(1,n_input).'  # weights (input-hidden) [hidden_neuron,input_neuron]
		w_0_cand[c] = rand()  # biases of each hidden neuron
		w_hh_cand[c] = [rand(1,n_hidden) 0].' # weights (hidden-hidden) [hidden_neuron_from]

		err_cand = 0	# correlation between output of hidden unit and residual output error of the network
						# (to decide which candidate unit is best)

		if (n_hidden == 0) # if no hidden units yet
			(w_cand[c],w_0_cand[c],w_hh_cand[c],err_cand) =
			adjust_hidden(n_input,n_hidden,0,0,0,0,v_0,w_io,training_set_in,training_set_out,alpha_hid_in,w_cand[c],w_0_cand[c],w_hh_cand[c])
		else
			(w_cand[c],w_0_cand[c],w_hh_cand[c],err_cand) =
			adjust_hidden(n_input,n_hidden,w,w_0,v,w_hh,v_0,w_io,training_set_in,training_set_out,alpha_hid_in,w_cand[c],w_0_cand[c],w_hh_cand[c])
		end

		if (err_cand < err_min)  # if this candidate is better
			w_best_cand = w_cand[c]
			w_0_best_cand = w_0_cand[c]
			w_hh_best_cand = w_hh_cand[c]
			err_min = err_cand
		end
	end

	n_hidden += 1

	if (n_hidden == 1)
		w = w_best_cand.'
		w_0 = w_0_best_cand
		w_hh = 0
		v = rand()
	else
		w = [w; w_best_cand.']
		w_0 = [w_0; w_0_best_cand]
		w_hh = [w_hh zeros(n_hidden-1,1)]
		w_hh = [w_hh; w_hh_best_cand.']
		v = [v; rand()]
	end

	return (w, w_0, w_hh, v, n_hidden, err_min)

end
