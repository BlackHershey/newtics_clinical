import re
import zipfile
import pandas as pd
import os
from glob import glob
from os import getcwd, listdir, path
from os.path import join, basename
from gooey import Gooey, GooeyParser


def score_drz_trainer(directory):
    os.chdir(directory)
    df = pd.read_csv(directory+"\\drz_output_copy.csv")
    for item in os.listdir(directory):
        if item.endswith(".zip"):
            with zipfile.ZipFile(item, 'r') as f:
                names = f.namelist()
                trainer = [ f for f in names if re.search(r'au', f, flags=re.IGNORECASE) ]
                if len(trainer) > 0:
                    subject, event_name = item[:-4].split("_")
                    trainer.sort() 
                    session_count = 1
                    for file in trainer:
                        print("Processing", item, file)
                        if file.startswith("Users"):
                            file = file.rstrip().split("/")[3]
                        if not path.exists(file):
                            file = file.rstrip()[:-5] + "_NOGOOD.ttsd"
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            condition = lines[0].rstrip().split("|")[2]
                            for line in lines:
                                if line.startswith("number of tics"):
                                    tics = line.rstrip().split("|")[1]
                                if line.startswith("number of 10s"):
                                    rewards = line.rstrip().split("|")[1]
                                if line.startswith("session length"):
                                    duration = float(line.rstrip().split("|")[1])/60
                            entry = {'demo_study_id': subject.upper(), 'event_name': event_name, "session": session_count, "condition": condition, "tics": tics, "rewards": rewards, "duration": duration,
                                    "file": join(directory, file)}
                            session_count += 1
                            df = df.append(entry, ignore_index=True)
    df.sort_values(by=['demo_study_id', 'event_name', 'session']).reset_index(drop=True)
    df.to_csv('drz_output_trainer.csv')




if __name__ == '__main__':
    # set up expected arguments and associated help text
    @Gooey()
    def parse_args():
        parser = GooeyParser(description='parses and scores drz files')

        required = parser.add_argument_group('Required Arguments', gooey_options={'columns':1})
        required.add_argument('--indir', widget='DirChooser', required=True, help='directory containing drz txt files')

        return parser.parse_args()


    args = parse_args()
    print(args)
    score_drz_trainer(args.indir)