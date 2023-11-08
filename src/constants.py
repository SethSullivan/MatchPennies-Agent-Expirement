import data_visualization as dv
from pathlib import Path
wheel = dv.ColorWheel()
def find_parent(path: Path, target_parent: str) -> Path | None:
    # `path.parents` does not include `path`, so we need to prepend it if it is
    # to be considered
    for parent in [path] + list(path.parents):
        if parent.name == target_parent:
            return parent


ELECTROMECHANICAL_DELAY = 50
xticklabel_colors_means = [wheel.rak_blue, wheel.rak_red, wheel.rak_orange,
                           wheel.dark_blue_hc, wheel.lighten_color(wheel.rak_red,1.5), wheel.burnt_orange,]

xticklabel_colors_sd = [wheel.rak_blue, wheel.dark_blue_hc, 
                        wheel.rak_red, wheel.lighten_color(wheel.rak_red,1.5),
                        wheel.rak_orange,wheel.burnt_orange,]

MODELS_PATH = find_parent(Path.cwd(),"MatchPennies-Agent-Expirement") / "results" / "models"
MODEL_INPUT_PATH = find_parent(Path.cwd(),"MatchPennies-Agent-Expirement") / "results" / "model_inputs"

model_input_savenames = ['exp1_rt_median_array.pkl','exp1_rt_sd_array.pkl',
                        'exp1_mt_median_array.pkl','exp1_mt_sd_array.pkl',
                        'exp1_timing_sd_array.pkl','exp1_agent_means.pkl','exp1_agent_sds.pkl']