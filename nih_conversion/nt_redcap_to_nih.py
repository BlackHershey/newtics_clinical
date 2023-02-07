import os
import pandas as pd
from redcap_to_nih import convert_redcap_to_nih
from gooey import Gooey, GooeyParser
from datetime import datetime

# These are subjects who were enrolled duirng the pre-R01 study, but also are enrolled in the R01 study and have data in the R01 study.
# We use their data from the "old" database for the demographics form.
PRE_R01_SUBJECTS = ['NT736', 'NT738', 'NT805', 'NT806', 'NT807', 'NT808', 'NT809', 'NT810', 'NT812', 'NT814', 'NT816', 'NT817']

fields_to_withhold = [ 'incl_excl_ic', 'incl_excl_who', 'incl_excl_new_tics_grp', 'share_data_permission', 'share_data_comments',
    'r01_survey_consent', 'demo_dob', 'childs_age', 'incl_excl_fon_scrn', 'dna_sample_lab_id', 'cbcl_birthdate', 'mo3fupc_who',
    r'\w*_data_files*$', 'age_at_visit', 'visit_referral_source' ]

def merge_projects(df1, df2):
	right_suffix = '_temp'
	merged_df = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=('', right_suffix))

	# use other study to fill in missing values in current study
	fill_cols = [ col[:-len(right_suffix)] for col in merged_df.columns if col.endswith(right_suffix) ]
	for col in fill_cols:
		merged_df[col] = merged_df[col].fillna(merged_df[col + right_suffix])

	# cleanup merge conflict columns
	other_cols = [ col for col in merged_df.columns if col.endswith(right_suffix) ]
	merged_df.drop(other_cols, axis=1, inplace=True)

	return merged_df

def merge_nt_redcap_dfs(nt_file, r01_file):
    # these are the fields from the "old" database that we need to merge with the new R01 database
    nt_fields = [
        'visit_date', 
        'demo_sex', 
        'demo_dob', 
        ]
    demo_fields = [ 
        'demo_childs_edu', 
        'demo_completed_by', 
        'demo_ethnicity', 
        'demo_mat_edu', 
        'demo_maternal_mari', 
        'demo_pat_edu', 
        'demo_patern_mari', 
        'demo_prim_lang', 
        'demo_race', 
        'demo_secondary_language', 
        'handedness', 
        ]
    nt_fields.extend(demo_fields)
    nt_df = pd.read_csv(nt_file, index_col=[0,1])
    # nt_df = nt_df[nt_fields]
    r01_df = pd.read_csv(r01_file, index_col=[0,1])

    # drop screen visit arm for pre-R01 subjects from R01 df (since we're getting that from the pre-R01 df)
    # AND keep only screen data from pre-R01 subjects in the old df
    drop_tuples = []
    keep_tuples = []
    for sub in PRE_R01_SUBJECTS:
        drop_tuples.append((sub, 'screening_visit_arm_1'))
        keep_tuples.append((sub, 'initial_screen_arm_1'))
    r01_drop_mask = r01_df.index.isin(drop_tuples)
    r01_df = r01_df[~r01_drop_mask]
    nt_keep_mask = nt_df.index.isin(keep_tuples)
    nt_df = nt_df[nt_keep_mask]

    # rename nt screen visits
    nt_df = nt_df.rename(index={'initial_screen_arm_1':'screening_visit_arm_1'})

    # make subjects NT736 through NT817 "NewTics" group
    nt_df['incl_excl_grp'] = 1

    # merge pre-R01 and R01 data
    merged_df = merge_projects(nt_df, r01_df)
    merged_df = merged_df.dropna(how='all')

    # remove rows for subjects in nt 9-11.5mo group
    nt9_11_ids = [ x[0] for x in merged_df[merged_df['incl_excl_new_tics_grp'] == 2].index.tolist() ]
    merged_df = merged_df.drop(nt9_11_ids, level='demo_study_id')

    # remove rows for excluded participants
    merged_df = merged_df[merged_df['incl_excl_eligible'] != 0]

    return merged_df

@Gooey()
def parse_args():
    parser = GooeyParser()
    required = parser.add_argument_group('Required Arguments')
    # required.add_argument('--guid_file', widget='FileChooser', required=True, help='GUID spreadsheet (xlsx)')
    # required.add_argument('--guid_password', widget='PasswordField', required=True, help='password for GUID spreadsheet')
    required.add_argument('--redcap_data_dictionary', widget='FileChooser', required=True, help='RedCap data dictionary (csv)')
    required.add_argument('--nih_dd_directory', widget='DirChooser', required=True, help='NIH data dictionary directory')
    required.add_argument('--form_mapping_key', widget='FileChooser', required=True, help='Form mapping key file (xlsx)')
    required.add_argument('--output_directory', widget='DirChooser', required=True, help='Output directory')

    input = parser.add_argument_group('Data Input')
    input.add_argument('--nt_data_file', widget='FileChooser', required=True, help='Data exported from NewTics pre-R01 RedCap project (csv)')
    input.add_argument('--r01_data_file', widget='FileChooser', required=True, help='Data exported from NewTics R01 RedCap project (csv)')

    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--item_level_replacements', widget='FileChooser', help='Item-level replacements (csv)')
    optional.add_argument('--to_date', widget='DateChooser', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='only process subjects up until date')
    optional.add_argument('--convert_forms', nargs='+', help='NIH form(s) to convert (default is all)')
    optional.add_argument('--redo', action='store_true', default=True, help='recreate import file even if already exists')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # combine pre-R01 and R01 RedCap data
    data_df = merge_nt_redcap_dfs(args.nt_data_file, args.r01_data_file)
    # data_df.to_csv(os.path.join(args.output_directory, 'nt_merged_df.csv'))

    # call convert_redcap_to_nih
    convert_redcap_to_nih( 
        data_df, 
        args.redcap_data_dictionary, 
        args.nih_dd_directory, 
        args.form_mapping_key, 
        args.output_directory, 
        args.item_level_replacements, 
        args.convert_forms, 
        fields_to_withhold, 
        args.to_date, 
        args.redo 
        )