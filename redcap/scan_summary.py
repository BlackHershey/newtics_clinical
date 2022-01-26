import argparse
import numpy as np
import os.path
import pandas as pd
import re
import utils

from getpass import getuser
from glob import glob

# BIDS_DIR = r'\\linux5\\data\cn6\soyoung\NewTics\BIDS'
BIDS_DIR = r'Z:\NewTics\BIDS'

def gen_participants_tsv(analysis_dir):
    bidsfile = os.path.join(BIDS_DIR, 'participants.tsv')

    df = pd.read_csv(os.path.join(analysis_dir, 'r01_result.csv'))
    df['demo_study_id'] = 'sub-' + df['demo_study_id']
    df = df.set_index('demo_study_id')[['age_screen', 'sex_screen', 'group_screen']]
    df.index.names = ['participant_id']
    df = df.rename(columns={col: col.split('_')[0] for col in df.columns}) # remove screen suffix from column names
    
    bids_subs = [ os.path.basename(f) for f in glob(os.path.join(BIDS_DIR, 'sub-*')) ]

    df = df[df.index.isin(bids_subs)]
    df.to_csv(bidsfile, sep='\t')
    return df


def summarize_scan_stats(outdir):
    bids_df = gen_participants_tsv(outdir)

    scan_visits = []
    files = glob(os.path.join(BIDS_DIR, 'sub-*', 'ses-*', '*_scans.tsv'))
    for f in files:
        sub, ses = re.search(r'(sub-\w+)_ses-(\w+)_', f).groups()
        if ses[-1] in ['2','3']: # hack to skip repeat scan day sessions 
            continue 
        scan_date = np.genfromtxt(f, usecols=[1], skip_header=1, dtype=np.datetime64)[0]
        scan_visits.append([sub, ses, scan_date])
    
    df = pd.DataFrame(scan_visits, columns=['participant_id', 'visit_type', 'scan_date']).set_index('participant_id')
    df['visit_type'] = df['visit_type'].replace('screen1', 'screen') # these should count as same visit
    df = bids_df.join(df)

    # save yearly scan stats to file
    r01_df = df[df['scan_date'] > np.datetime64('2017-07-01')]
    r01_df['year'] = r01_df['scan_date'].dt.year
    yearly_starts = r01_df.groupby(['year', 'visit_type', 'group'])['scan_date'].count()
    yearly_starts.to_csv(os.path.join(outdir, 'r01_scan_stats_by_year.csv'))
    
    # save whole sample stats to file
    res = df.groupby(['visit_type', 'group'])['scan_date'].count()
    res.to_csv(os.path.join(outdir, 'r01_scan_stats.csv'))


if __name__ == '__main__':
    _, study_vars = utils.get_study_vars('r01')
    basedir = study_vars['directories']['base'].format(getuser())
    outdir = study_vars['directories']['output'].format(basedir)

    summarize_scan_stats(outdir)