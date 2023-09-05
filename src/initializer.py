from pathlib import Path
import os

# Initial thangs
class InitialThangs:
    def __init__(self, experiment):
        self.experiment = experiment
        os.chdir(f'D:\OneDrive - University of Delaware - o365\Subject_Data\MatchPennies_Agent_{experiment}')
        self.PATH = Path(os.getcwd())
        self.SAVE_PATH = Path(f'C:\\Users\\Seth Sullivan\\OneDrive - University of Delaware - o365\\Desktop\\MatchPennies-Agent-Expirement\\Group_Figures\\{experiment}\\')
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

        # Fields pull and pull list
        self.figures_pull_list = []
        self.fields_pull = []
        with open(self.PATH / "Figures_Pull_List.txt",'r') as pull_file:
            self.figures_pull_list = pull_file.read().splitlines()
        with open(self.PATH/"Fields_Pull.txt",'r') as self.fields_pull:
            self.fields_pull = self.fields_pull.read().splitlines()
        self.num_subjects = len(self.figures_pull_list)

        if experiment == 'Exp1':
            self.tp3_title = "1000 [50]"
            self.tp4_title = "1000 [150]"
            self.tp5_title = '1100 [50]'
            self.tp6_title = "1100 [150]"
            self.tp7_title = "1200 [50]"
            self.tp8_title = '1200 [150]'
            self.trial_block_titles = [self.tp3_title, self.tp4_title, self.tp5_title,self.tp6_title, self.tp7_title, self.tp8_title]
            self.xlabel = 'Agent Mean [SD] Movement Onset Time (ms)'
            self.num_blocks = len(self.trial_block_titles)
            self.num_trials = 80
            self.num_rows,self.num_cols = 2,3
            self.condition_nums = ['0','1','2','3','4','5']
            self.f1_xlabel = 'Agent Mean Movement Onset Time (ms)'
            self.f2_xlabel = 'Agent SD Movement Onset Time (ms)'
            self.f1_collapse_xticklabs = ['1000','1100','1200']
            self.f2_collapse_xticklabs = ['50','150']
            
        if experiment == 'Exp2':
            self.tp3_title = "Win = 1\nIncorrect = 0\nIndecision = 0"
            self.tp4_title = "Win = 1\nIncorrect = -1\nIndecision = 0"
            self.tp5_title = "Win = 1\nIncorrect = 0\nIndecision = -1"
            self.tp6_title = "Win = 1\nIncorrect = -1\nIndecision = -1"
            self.trial_block_titles = [self.tp3_title, self.tp4_title, self.tp5_title,self.tp6_title]
            self.num_blocks = len(self.trial_block_titles)
            self.xlabel = 'Feedback Condition'
            self.condition_nums = ['0','1','2','3']

            self.num_trials = 80
            self.num_rows,self.num_cols = 2,2
            self.f1_xlabel = 'Incorrect Reward'
            self.f2_xlabel = 'Indecision Reward'
            self.f1_collapse_xticklabs = ['0 Incorrect','-1 Incorrect']
            self.f2_collapse_xticklabs = ['0 Indecision','-1 Indecision']