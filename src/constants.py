import data_visualization as dv
wheel = dv.ColorWheel()

ELECTROMECHANICAL_DELAY = 50
xticklabel_colors_means = [wheel.rak_blue, wheel.rak_red, wheel.rak_orange,
                           wheel.dark_blue_hc, wheel.lighten_color(wheel.rak_red,1.5), wheel.burnt_orange,]

xticklabel_colors_sd = [wheel.rak_blue, wheel.dark_blue_hc, 
                        wheel.rak_red, wheel.lighten_color(wheel.rak_red,1.5),
                        wheel.rak_orange,wheel.burnt_orange,]