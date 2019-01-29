import csv
import os
import pandas as pd

import sys
sys.path.append('../../../../NewTics/Data/analysis/clinical_data/scripts')

from getpass import getpass
from redcap_to_nih import get_redcap_df

study_dir = r'B:\NewTics'

def merge_image03():
    image03_csv = os.path.join(study_dir, 'scripts', 'image03_nodemo.csv')
    image_df = pd.read_csv(image03_csv)

    guid_df = pd.read_csv('guids.csv', index_col=1).rename(columns={'GUID': 'subjectkey'}).drop(columns='Date')
    guid_df.index.names = ['demo_study_id']
    demo_df = get_redcap_df(guid_df)[['demo_sex', 'demo_dob', 'subjectkey']].reset_index()
    demo_df = demo_df[demo_df['redcap_event_name'] == 'screening_visit_arm_1']

    image_df = image_df.merge(demo_df, on='demo_study_id')
    image_df['interview_date'] = pd.to_datetime(image_df['interview_date'], format='%Y%m%d')
    image_df['interview_age'] = (image_df['interview_date'] - pd.to_datetime(image_df['demo_dob'])).apply(lambda x: round(.0328767*x.days) if pd.notnull(x) else np.nan)
    image_df['interview_date'] = image_df['interview_date'].map(lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else x)
    image_df = image_df.drop(columns=['demo_dob', 'redcap_event_name'])
    image_df = image_df.rename(columns={'demo_sex': 'gender'})
    #image_df['image_file'] = image_df['image_file'].str.replace('..', study_dir)

    with open('image03.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', '3'])

    image_df.to_csv('image03.csv', mode='a', index=False, float_format='%g')

if __name__ == '__main__':
    merge_image03()
