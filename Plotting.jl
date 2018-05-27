# Plots decision values for grid -10 to 10 with step 0.1 for x and y

function plot_decision_boundary(nn_model, err_arr)

  x_plot = 0:0.1:10
  y_plot = 0:0.1:10

  hidden(x,y) = begin
    feedforward([x,y], nn_model)[2][1]
  end
  xgrid = repmat(x_plot',length(y_plot),1)
  ygrid = repmat(y_plot,1,length(x_plot))
  z = map(hidden, xgrid, ygrid)

  fig = figure("surf_plot")
  #ax = fig[:add_subplot](2,1,1)
  #subplot(211)
  #ax[:plot_surface](xgrid, ygrid, z)
  surf(xgrid, ygrid, z)
  xlabel("X")
  ylabel("Y")
  title("Surface Plot")
  show()
  #display(fig)

  #subplot(212)
  #fig2 = figure("err")
  #plot(collect(1:length(err_arr)), err_arr)
  #display(fig2)

  #subplot(212)
  #ax = fig[:add_subplot](2,1,2)
  #cp = ax[:contour](xgrid, ygrid, z)
  #ax[:clabel](cp, inline=1, fontsize=10)
  #xlabel("X")
  #ylabel("Y")
  #title("Contour Plot")
  #tight_layout()

  #for i=1:size(training_set_in,1)
  #  if (training_set_out[i] == 1)
  #    plot(training_set_in[i,1],training_set_in[i,2],"rx")
  #  else
  #    plot(training_set_in[i,1],training_set_in[i,2],"bo")
  #  end
  #end

  #cs = contour(xgrid,ygrid,z,fill=true)
  #colorbar(cs, shrink=0.8, extent='both')

  #surf(x_plot, y_plot, z_plot_delta)
  #surf(x_plot, y_plot, z_plot_hidden)
  #plot3D(training_set_in[:,1], training_set_in[:,2], training_set_out, marker='.')

  #return fig

end

# Plot data points
function plot_data(training_set_in, training_set_out)
    scatter(training_set_in[training_set_out .== 1.0, 1],
        training_set_in[training_set_out .== 1.0, 2],
        s=nothing,
        c=nothing,
        marker="o")
    scatter(training_set_in[training_set_out .== -1.0, 1],
        training_set_in[training_set_out .== -1.0, 2],
        s=nothing,
        c=nothing,
        marker="x")
    xlim((0, 10))
    ylim((0, 10))
end
