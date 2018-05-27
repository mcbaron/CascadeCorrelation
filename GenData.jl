# Generate data for testing

function gen_data(n_points, which)

    # Circular cluster of data
    if which == "circular"
        # Limits
        x_lim = [0, 10]
        y_lim = [0, 10]
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 10
            if abs( (x_arr[i,1] - 5.0)^2 + (x_arr[i,2] - 5.0)^2 ) > 3^2
                y_arr[i] = -1.0
            else
                y_arr[i] = 1.0
            end
        end

    # Linear separation
    elseif which == "linear"
        # Limits
        x_lim = [0, 10]
        y_lim = [0, 10]
        # Arguments and labels
        x_arr = zeros(n_points,2)
        y_arr = zeros(n_points)
        # Generation of each point
        for i=1:n_points
            x_arr[i,:] = rand(2)' * 10
            if x_arr[i,2] > 8.0 - 0.8 * x_arr[i,1]
                y_arr[i] = -1.0
            else
                y_arr[i] = 1.0
            end
        end
    end

    return x_arr, y_arr

end
