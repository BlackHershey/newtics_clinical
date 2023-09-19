import os
import pandas as pd
from redcap_to_nih import convert_redcap_to_nih
from gooey import Gooey, GooeyParser
from datetime import datetime

fields_to_withhold = [ 'incl_excl_ic', 'incl_excl_who', 'incl_excl_new_tics_grp', 'share_data_permission', 'share_data_comments',
    'r01_survey_consent', 'demo_dob', 'childs_age', 'incl_excl_fon_scrn', 'dna_sample_lab_id', 'cbcl_birthdate', 'mo3fupc_who',
    r'\w*_data_files*$', 'age_at_visit', 'visit_referral_source' ]


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
    input.add_argument('--redcap_data_file', widget='FileChooser', required=True, help='Data exported from LoTS RedCap project (csv raw)')

    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--item_level_replacements', widget='FileChooser', help='Item-level replacements (csv)')
    optional.add_argument('--to_date', widget='DateChooser', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='only process subjects up until date')
    optional.add_argument('--convert_forms', nargs='+', help='NIH form(s) to convert (default is all)')
    optional.add_argument('--redo', action='store_true', default=True, help='recreate import file even if already exists')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # Open RedCap data file as pandas df
    redcap_data_df = pd.read_csv(args.redcap_data_file, index_col=[0,1])

    # call convert_redcap_to_nih
    convert_redcap_to_nih( 
        redcap_data_df, 
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