import data_visualization as dv
from pathlib import Path
import numpy as np
wheel = dv.ColorWheel()
def find_parent(path: Path, target_parent: str) -> Path | None:
    # `path.parents` does not include `path`, so we need to prepend it if it is
    # to be considered
    for parent in [path] + list(path.parents):
        if parent.name == target_parent:
            return parent


ELECTROMECHANICAL_DELAY = 50
xticklabel_colors_means = [wheel.rak_blue, wheel.rak_orange, wheel.rak_red,
                           wheel.dark_blue_hc,wheel.burnt_orange,wheel.lighten_color(wheel.rak_red,1.5)]

xticklabel_colors_sd = [wheel.rak_blue, wheel.dark_blue_hc, 
                        wheel.rak_orange,wheel.burnt_orange,
                        wheel.rak_red, wheel.lighten_color(wheel.rak_red,1.5),
                        ]
collapse_colors = [
    wheel.blend_colors(xticklabel_colors_means[0],xticklabel_colors_means[3]),
    wheel.blend_colors(xticklabel_colors_means[1],xticklabel_colors_means[4]),
    wheel.blend_colors(xticklabel_colors_means[2],xticklabel_colors_means[5])
]
# xticklabels = ["(1000, 50)", "(1100, 50)", "(1200, 50)", "(1000, 150)", "(1100, 150)", "(1200, 150)"]
# xticklabels = [r"Early $\mathbf{\mu}$"+"\n"+r"Low $\mathbf{\sigma}$", 
#                r"Middle $\mathbf{\mu}$"+"\n"+r"Low $\mathbf{\sigma}$", 
#                r"Late $\mathbf{\mu}$"+"\n"+r"Low $\mathbf{\sigma}$",
#                r"Early $\mathbf{\mu}$"+"\n"+r"High $\mathbf{\sigma}$", 
#                r"Middle $\mathbf{\mu}$"+"\n"+r"High $\mathbf{\sigma}$", 
#                r"Late $\mathbf{\mu}$"+"\n"+r"High $\mathbf{\sigma}$"]
xticklabels = [r"Early Mean"+"\n"+r"Low Variance", 
               r"Middle Mean"+"\n"+r"Low Variance", 
               r"Late Mean"+"\n"+r"Low Variance",
               r"Early Mean"+"\n"+r"High Variance", 
               r"Middle Mean"+"\n"+r"High Variance", 
               r"Late Mean"+"\n"+r"High Variance"]
xticklabels_sd = [r"Early Mean"+"\n"+r"Low Variance", 
               r"Early Mean"+"\n"+r"High Variance", 
               r"Middle Mean"+"\n"+r"Low Variance",
               r"Middle Mean"+"\n"+r"High Variance", 
               r"Late Mean"+"\n"+r"Low Variance", 
               r"Late Mean"+"\n"+r"High Variance"]
MODELS_PATH = find_parent(Path.cwd(),"MatchPennies-Agent-Expirement") / "results" / "models"
MODEL_INPUT_PATH = find_parent(Path.cwd(),"MatchPennies-Agent-Expirement") / "results" / "model_inputs"

model_input_savenames = ['exp1_rt_median_array.pkl','exp1_rt_sd_array.pkl',
                        'exp1_mt_median_array.pkl','exp1_mt_sd_array.pkl',
                        'exp1_timing_sd_array.pkl','exp1_agent_means.pkl','exp1_agent_sds.pkl']
# This goes along with parameter distribution so I know which index corresponds to which metric
input_keys = ["rt","rt_sd","mt","mt_sd","timing_sd",]

stimulus_means = np.array([1000.0,1000.0,1100.0,1100.0,1200.0,1200.0])
stimulus_sds = np.array([50.0,150.0,50.0,150.0,50.0,150.0])

# model_colors = [wheel.grey, wheel.lighten_color(wheel.dark_grey,1.25), wheel.black]
battleship_grey = "#848482"
charcoal        = "#36454F"
stone_grey      = "#928E85"
cool_grey       = "#9090C0"
cadet_grey      = "#91A3B0"
marengo         = "#4C5866"
pebble          = "#333333"
model_colors = [cadet_grey, wheel.lighten_color(pebble,0.75), wheel.black]
model_names = ["No Switch Time Model", "Known Switch Time Model", "Unknown Switch Time Model"]

stimulus_color = wheel.lighten_color(wheel.yellow, 1.75)
player_color = wheel.purple
