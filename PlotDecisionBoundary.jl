# Plots decision values for grid -10 to 10 with step 0.1 for x and y

function plot_decision_boundary()

  #x_plot = linspace(0.1,7,70)
  x_plot = -10:0.1:10
  #y_plot = linspace(0.1,7,70)
  y_plot = -10:0.1:10
  #z_plot_delta = zeros(70,70)
  #z_plot_hidden = zeros(70,70)

  #for i=1:1:70
  #  for j=1:1:70
  #    z_plot_hidden[i,j] = feedforward([x_plot[i] y_plot[j]],n_input,w,w_0,size(w,1),v,v_0,w_hh,w_io)[2][1]
  #    z_plot_delta[i,j] = feedforward([x_plot[i] y_plot[j]],n_input,0,0,0,0,v_0,0,w_io)[2][1]
  #  end
  #end

  hidden(x,y) = begin
    feedforward([x y],n_input,w,w_0,size(w,1),v,v_0,w_hh,w_io)[2][1]
  end
  xgrid = repmat(x_plot',length(y_plot),1)
  ygrid = repmat(y_plot,1,length(x_plot))
  z = map(hidden,xgrid,ygrid)

  fig = figure("surf_plot")
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](xgrid, ygrid, z)
  xlabel("X")
  ylabel("Y")
  title("Surface Plot")

  subplot(212)
  ax = fig[:add_subplot](2,1,2)
  cp = ax[:contour](xgrid, ygrid, z)
  #ax[:clabel](cp, inline=1, fontsize=10)
  xlabel("X")
  ylabel("Y")
  title("Contour Plot")
  tight_layout()

  #cs = contour(xgrid,ygrid,z,fill=true)
  #colorbar(cs, shrink=0.8, extent='both')

  #surf(x_plot, y_plot, z_plot_delta)
  #surf(x_plot, y_plot, z_plot_hidden)
  #plot3D(training_set_in[:,1], training_set_in[:,2], training_set_out, marker='.')

end
