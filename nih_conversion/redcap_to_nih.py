import csv
import numpy as np
import os.path
import pandas as pd
import re
import win32com.client

import sys
sys.path.append('..')

from datetime import datetime
from enum import Enum
from getpass import getuser, getpass
from gooey import Gooey, GooeyParser
from itertools import chain

BASE_PATH = r'C:\Users\{}\Box\Black_Lab\projects\TS\New_Tics_R01\Data'.format(getuser())
CONVERSION_DIR = os.path.join(BASE_PATH, 'NDA', 'conversion')
# DATA_DICT_PATH = os.path.join(BASE_PATH, 'REDCap', 'REDCap_data_dictionaries', 'NewTicsR01_DataDictionary_2019-10-15.csv')
# FORM_MAP_PATH = os.path.join('cfg', '2692.MappingKey.xlsx')
# GUID_PATH = os.path.join(BASE_PATH, r'NIMH GUID\GUIDs.xlsx')
# OUTDIR = os.path.join(CONVERSION_DIR, 'import_forms')

CHECKBOX_TO_LABEL = [ 'incl_excl_concom_meds', 'demo_race'] # checkbox fields that will need to expanded into concatenated label string
OTHER_LABELS_NEEDED = [ r'demo_\w*_mari', r'srs_800_q\d+', 'pedsql_version', 'ksads5_asd_specify' ] # other fields to concatenate label strings

NUMBER_PATTERN = r'(\d+(?:\.\d+)?)' # regex for numbers only string (both int and float)

class AgeUnits(Enum):
    YEARS = 0
    MONTHS = 1
    WEEKS = 2
    DAYS = 3

"""
Get all columns that match a regex pattern
"""
def get_matching_cols(df, pattern):
    return [ col for col in df.columns if re.match(pattern, col) ]


"""
Get map of options for radio button/checkbox variable (key = value stored in data, value = human-readable label)
"""
def get_data_dict_options_map(variable, choices_str):
    # print('CHECKBOX VARIABLE = {}'.format(variable))
    # print('CHOICES STRING = {}'.format(choices_str))
    # choices_str = data_dict_df.xs(variable)['choices']
    options = [ choice.split(',', 1) for choice in choices_str.split('|') ]
    options = [ opt for opt in options if opt != ['']] # handle extra '|' at end of choice specification
    return { int(option[0].strip()): option[-1].strip() for option in options }

"""
Concatenate multiple value response into one string with specified delimiter
"""
def concat_column_values(row, sep=';'):
    return row[row.notnull()].astype(str).str.cat(sep=sep) if pd.notnull(row).any() else np.nan


"""
Consolidate checkbox responses into one variable
(Individual checkboxes for a variable are stored in separate columns with ___# suffix)
"""
def replace_checkbox_with_label(df, variable, choices_str, use_label=False):
    num_to_label_map = get_data_dict_options_map(variable, choices_str)
    checkbox_cols = get_matching_cols(df, variable + r'___\d+') # get all columns that correspond to variable
    for col in checkbox_cols:
        checkbox_num = re.search(r'___(\d+)', col).group(1) 
        replace1 = num_to_label_map[int(checkbox_num)] if use_label else checkbox_num # optionally replace number with label 
        df[col] = df[col].replace([0, 1], [np.nan, replace1]) # swap in expected values
        df[col] = df[col].replace('None of the above', np.nan)

    df[variable] = df[checkbox_cols].apply(concat_column_values, axis=1) # consolidate responses 
    df = df.drop(columns=checkbox_cols) # drop the separate columns 
    return df


"""
Get label for radio button selected response
"""
def replace_num_with_label(df, variable, choices_str):
    num_to_label_map = get_data_dict_options_map(variable, choices_str)
    df[variable] = df[variable].replace(num_to_label_map)
    return df


"""
Convert reported ages to expected units
    Parses free-text strings (i.e. 1yr 2mo) to extract number associated with each unit, then calculates response in desired units
Input params:
    row - row of dataframe 
    dest_unit - unit that we want to the variable to be in
    src_unit - optional, unit that we collect the variable in (if different from NIH-desired unit)

"""
def age_to_units(row, dest_unit, src_unit=None):
    src_unit = dest_unit if not src_unit else src_unit # if no src_unit, assume we collect it in the same units NIH expects
    age_searches = [ 'year|yr|y', 'month|mo|m', 'week|wk|w', 'day|d' ] # possible abbreviations for time units
    for col in row.axes[0].tolist():
        yrs_mos_wks_days = [0, 0, 0, 0] # create array to keep track of parsed units
        
        # if the response is just a number, set parsing array of src unit to that number
        no_units = re.search(NUMBER_PATTERN +'$', str(row[col]))
        if no_units:
            yrs_mos_wks_days[src_unit.value] = float(no_units.group(1))
        # otherwise, for each possible unit string extract the number associated with it and store it in the parsing array
        else:
            for i, search_str in enumerate(age_searches):
                res = re.search('(?:' + NUMBER_PATTERN + ' *(?:' + search_str +')s*)', str(row[col]), flags=re.IGNORECASE)
                yrs_mos_wks_days[i] = float(res.group(1)) if res else yrs_mos_wks_days[i]

        # based on the desired units, calculate value in desired units using all units in response
        if dest_unit == AgeUnits.YEARS:
            age = yrs_mos_wks_days[0] + (yrs_mos_wks_days[1] / 12) + (yrs_mos_wks_days[2] / 52) + (yrs_mos_wks_days[3] / 365) 
        elif dest_unit == AgeUnits.MONTHS:
            age = (yrs_mos_wks_days[0] * 12) + yrs_mos_wks_days[1] + (yrs_mos_wks_days[2] / 4.345) + (yrs_mos_wks_days[3] /  30.417)
        elif dest_unit == AgeUnits.WEEKS:
            age = (yrs_mos_wks_days[0] * 52) + (yrs_mos_wks_days[1] * 4.345) + yrs_mos_wks_days[2] + (yrs_mos_wks_days[3] / 7)
        elif dest_unit == AgeUnits.DAYS:
            age = (yrs_mos_wks_days[0] * 365) + (yrs_mos_wks_days[1] * 30.417) + (yrs_mos_wks_days[2] * 7) + yrs_mos_wks_days[3]

        row[col] = round(age) if age != 0 else np.nan
    return row


def mom_or_dad(response):
    try:
        if re.match('^(grandma|grandmother|grandpa|grandfather)', response, flags=re.IGNORECASE):
            return 3
        elif re.match('^.*(mom|mother)', response, flags=re.IGNORECASE):
            return 1
        elif re.match('^.*(dad|father)', response, flags=re.IGNORECASE):
            return 2
        else:
            return 999
    except:
        return 999

def respond_code(response):
    try:
        if re.match('^(grandma|grandmother|grandpa|grandfather)', response, flags=re.IGNORECASE):
            return 4
        elif re.match('^.*(mom|mother|dad|father)', response, flags=re.IGNORECASE):
            return 1
        else:
            return 999
    except:
        return 999


"""
Handle setting SES for divorced parents
    NIH only allows specification of one parent's partner's info (should be that of the subject's primary residence) but we collect both
    Figure out who the primary residence is and then fill in their partner's info
"""
def ses_primary_partner(row):
    spouse_cols = [ 'ses_edu_level', 'ses_occ' ]
    mpartner_cols = [ col + '_mpartner' for col in spouse_cols ]
    fpartner_cols = [ col + '_fpartner' for col in spouse_cols ]

    if row[mpartner_cols + fpartner_cols].isnull().all():
        return row
    # print('mom_or_dad input = {}: {}'.format(row['demo_study_id'], row['ses_primary_res']))
    try:
        primary_residence = mom_or_dad(row['ses_primary_res'])
    except:
        primary_residence = ''
    if primary_residence == 1:
        row['version_form'] = 'Mother\'s partner'
        row[['bsmss03_spouse', 'bsmss07_spouse']] = row[mpartner_cols]
    elif primary_residence == 2:
        row['version_form'] = 'Father\'s partner'
        row[['bsmss03_spouse', 'bsmss07_spouse']] = row[fpartner_cols]
    else:
        row['version_form'] = ''
        row[['bsmss03_spouse', 'bsmss07_spouse']] = ['','']

    return row

"""
Convert sparse int column
    Some variables that contain ints for some records but NaN for others are hard to write to CSV as int.
"""
def sparse_int_fix(x):
    try:
        if '.' in x:
            return x.split('.')[0]
        else:
            return ''
    except:
        return ''


def add_years_str(row):
    return row + ' years' if str(row).isdigit() else row


"""
Break up concatenated variables
    Certain variables (i.e. race) we allow multiple responses but NIH doesn't 
    (they want one response in the main variable and then others in an "additional details" overflow columns)
"""
def separate_multi_response(row, response_col, other_col=None, sep=';'):
    if pd.notnull(row[response_col]):
        responses = row[response_col].split(sep)
        row[response_col] = responses[0]
        row[other_col] = responses[1:] if (other_col and len(responses) > 1) else np.nan
    return row


"""
Split up the raw and Tscores by sex
    NIH has separate columns for male/female norms
"""
def sex_norm_srs(df, cols, type):
    rename_dict = { col: type + str(i+1) for i, col in enumerate(cols[:5]) }
    total_col = 'srs_total{}'.format('_t' if type == 'tscore' else '')
    rename_dict[total_col] = '{}{}all'.format(type, 'score' if type == 'raw' else '')
    df = df.rename(columns=rename_dict)

    renamed_cols = list(rename_dict.values())
    for sex in [('M', 'male'), ('F', 'female')]:
        new_cols = [ '_'.join([sex[1], col]) for col in renamed_cols ]
        for i, col in enumerate(new_cols):
            df[col] = np.where(df['sex'] == sex[0], df[renamed_cols[i]], np.nan)
        score_all_col = [ col for col in new_cols if col.endswith('scoreall') ][0]
        df.loc[df['sex'] == sex[0], score_all_col] = df.loc[df['sex'] == sex[0], score_all_col].fillna(999)
    return df.drop(columns=renamed_cols)


"""
Read in password-protected Excel file that contains NTID/GUID mapping
"""
def get_guid_df(guid_file, guid_pw):
    xlApp = win32com.client.Dispatch('Excel.Application')
    xlws = xlApp.Workbooks.Open(guid_file, False, True, None, guid_pw).Sheets(1)
    content = list(xlws.Range(xlws.Cells(2, 2), xlws.Cells(1000,3)).Value)
    df = pd.DataFrame(content, columns=['demo_study_id', 'subjectkey']).dropna()
    return df.set_index('demo_study_id')


def decrement_scale(df):
    return df.select_dtypes(['int']) - 1


"""
ADHD RS updates -- shared between all form versions
    Add form metadata to ADHD RS (multi-form; we collect as separate forms, NIH has separate row for each form)
"""
def update_adhdrs(temp_df, form):
    who = 'expert' if 'expert' in form else 'parent'
    which = 'worst-ever' if 'lifetime' in form else 'past week'
    temp_df['version_form'] = '; '.join([who, which])
    return temp_df


"""
CYBOCS updates -- shared between all form versions
    Add form metadata to CYBOCS (multi-form; we collect as separate forms, NIH has separate row for each form)
    Choose most specific response from checkbox fields where we allow multiple but NIH doesn't (changes here because form-version specific)
"""
def update_cybocs(temp_df, form):
    if form in ['expert_cybocs_symptoms', 'cybocs_12mo']:
        temp_df = temp_df.replace('1;2', '1') # for followup, 1 (past week) is more specific than 2 (since last visit)
        temp_df = temp_df.replace(['2', '1'], ['4', '2']) # need to remap past week to '2' and create code '4' for since last visit
        temp_df = temp_df.replace(['1;3', '2;3'], ['1', '2']) # if never and some other option are checked, assume the other option
    elif form == 'childrens_yalebrown_oc_scale_self_report':
            temp_df = temp_df.replace('1;2', '2') # for lifetime form, 'when OCD was worst' (2) is more specific than 'ever in lifetime' (1)
    who = 'expert' if 'expert' in form else 'parent'
    which = ' '.join(form.split('_')[-2:]) if form in ['expert_cybocs_past_week', 'cybocs_worst_ever'] else None
    temp_df['version_form'] = '; '.join(filter(None, [who, which]))
    return temp_df


"""
TSCL updates -- shared between all form versions
    Add form metadata to TSCL (multi-form; we collect as separate forms, NIH has separate row for each form)
    Choose most specific response from checkbox fields where we allow multiple but NIH doesn't
"""
def update_ticscreener(temp_df, form):
    if form == 'tic_symptom_checklist_screening':
        temp_df = temp_df.replace('1;2', '2') # for screening checklist, 2 (past week) is more specific than 1 (lifetime)
    else:
        temp_df = temp_df.replace('1;2', '1') # for exp/followup, 1 (past week) is more specific than 2 (since last visit)
        temp_df = temp_df.replace(['2', '1'], ['4', '2']) # need to remap past week to '2' and create code '4' for since last visit

    temp_df = temp_df.replace(['1;3', '2;3'], ['1', '2']) # if never and some other option are checked, assume the other option

    temp_df['version_form'] = 'expert' if 'expert' in form else 'parent'
    return temp_df


"""
YGTSS updates -- shared between all form versions
    Add form metadata to YGTSS (multi-form; we collect as separate forms, NIH has separate row for each form)
"""
def update_ygtss(temp_df, form):
    temp_df['version_form'] = 'past week' if 'past_week' in form else 'post-drz'
    return temp_df


"""
Handle splitting of multiform rows into separate rows
    We collect multiple versions of forms from subjects at the same timepoint (i.e. in one row, but NIH wants them to have their own row)
    We also split some forms up into screen/12mo that contain the same data with different variable names (i.e. suffixed with _12mo)
    
Input params:
    form_dd_df - REDCap data dictionary dataframe for form
    nih_form - NIH form shortname
    form_df - dataframe with responses for particular form
    update_func - form-specific function to call after splitting (i.e. update_ygtss, update_adhdrs, etc.)
"""
def split_multiform_row(form_dd_df, nih_dd_directory, nih_form, form_df, update_func):
    nih_dd_df = pd.read_csv(os.path.join(nih_dd_directory, nih_form + '_definitions.csv'), usecols=['ElementName', 'Aliases'])     # read in NIH data dictionary
    result = None
    redcap_forms = form_dd_df['form'].unique() # get all REDCap forms that map to NIH form
    for form in redcap_forms:
        form_cols = [ col for col in form_df if col in form_dd_df[form_dd_df['form'] == form].index ] # get all columns for a REDCap form
        # print('\t', form, len(form_cols))
        temp_df = form_df[form_cols + [ col for col in form_df if col not in form_dd_df.index.values ]]
        temp_df = temp_df.dropna(how='all', subset=form_cols) # remove rows that are all null for current pattern (helps with screen/12mo form split)
        for col in form_cols:
            if col in nih_dd_df['ElementName'].values:
                continue
            try:
                # rename conflicting column to the NIH variable name
                temp_df = temp_df.rename(columns={col: nih_dd_df[nih_dd_df['Aliases'].str.contains(col, na=False)].iloc[0,0]})
            except Exception as e:
                print(col)
                raise e

        # update df if specfied
        if update_func:
            temp_df = update_func(temp_df, form) # update done here because we need to know the redcap form name
        temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()] # remove duplicate column names (see first comment: https://stackoverflow.com/a/27412913)
        result = pd.concat([result, temp_df], sort=False) if result is not None else temp_df # concatenate along row-axis 
    return result

"""
Replace study staff/doctor names with generic references
    NIH considers our info as PII
"""
def replace_staff_names(df):
    # ignore empy columns and the 'subjectkey' (NDAR GUID) column
    ignore_cols = [col for col in df.columns if df[col].isnull().all()]
    ignore_cols.append('subjectkey')
    subset = [ col for col in df.columns if col not in ignore_cols]

    # create search string
    staff_initials = [ 'ECB', 'VM', 'SK', 'KJB', 'SR', 'JH', 'BS', 'JA', 'AA', 'SG', 'LAG', 'NYY' ]
    staff_names = [ ('Emily', 'Bihun'), ('Vicki', 'Martin'), ('Soyoung', 'Kim'), ('Kevin', 'Black'), ('Samantha', 'Ranck'), ('Jackie', 'Hampton'), ('Jaridd', 'Albert'), ('Amanda', 'Arbuckle'), ('Sarah', 'Grossen'), ('Luis', 'Gonzalez'), ('Nancy', 'Yang') ]
    staff_name_matches = [ '{}(?: {})?'.format(first, last) for first, last in staff_names ] # create regexes of first name with optional last name
    staff_match_str = '|'.join(staff_initials + staff_name_matches)

    # replace matches in dataframe
    df[subset] = df[subset].replace(to_replace={'({})'.format(staff_match_str): 'rater', '(Dr. ([A-Z][a-z]+){1,2})': 'doctor'}, regex=True)

    return df

"""
Format date string in NIH-expected date format
"""
def format_date_str(date_series):
    return date_series.map(lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else x)

"""
Convert RedCap data df to NIH csv format
"""
def convert_redcap_to_nih(data_df, redcap_data_dictionary, nih_dd_directory, form_mapping_key, output_directory, item_level_replacements=None, convert_forms=None, fields_to_withhold=None, to_date=None, redo=False):
    # data_df.to_csv('data_df_before_convert.csv')

    # rename guid field to subjectkey
    guid_df = data_df[['guid']]
    guid_df = guid_df.dropna().rename(columns={'guid': 'subjectkey'})
    guid_df = guid_df.reset_index().set_index('demo_study_id').drop(columns='redcap_event_name')
    # guid_df.to_csv(os.path.join(output_directory, 'guid_df.csv'))

    # drop LoTS ineligible
    if 'incl_excl_eligible' in data_df.columns and 'mo3fupc_date' not in data_df.columns:
        # fill incl_excl_eligible across all visits
        data_df[['incl_excl_eligible']] = data_df[['incl_excl_eligible']].apply(lambda x: x.ffill().bfill())
        # select data only from eligible participants
        data_df = data_df[(data_df['incl_excl_eligible'] == 1)]

    # merge in GUID
    data_df = data_df.reset_index().join(guid_df, on='demo_study_id', how='left')

    # drop data for subjects with no GUID
    data_df = data_df[data_df['subjectkey'].str.contains("NDA", na = False)]

    data_dict_df = pd.read_csv(redcap_data_dictionary, index_col=0)
    data_dict_df = data_dict_df.rename(columns={'Form Name': 'form', 'Field Type': 'type', 'Choices, Calculations, OR Slider Labels': 'choices'})
    data_dict_df = data_dict_df[['form', 'type', 'choices']]

    form_map_df = pd.read_excel(form_mapping_key, skiprows=[1,2], index_col=0, usecols=[1,2])
    nih_forms = np.unique(form_map_df.index.values)
    if convert_forms:
        nih_forms = [ form for form in convert_forms if form in nih_forms ]

    # merge guid into data_df

    # convert sex codes & calculate interview age
    # LoTS doesn't have groups, change this to check for group variable
    if 'incl_excl_grp' in data_df.columns:
        # NewTics
        # data_df.to_csv('data_df_before_groupby.csv')
        # print(data_df.index)
        # new_df = data_df.groupby('demo_study_id', as_index=False)[['demo_sex', 'demo_dob', 'incl_excl_grp']].apply(lambda x: x.ffill().bfill()).reset_index(level=0, drop=True)
        # new_df.to_csv('data_df_after_groupby.csv')
        # print(new_df.index)
        data_df[['demo_sex', 'demo_dob', 'incl_excl_grp']] = data_df.groupby('demo_study_id', as_index=False)[['demo_sex', 'demo_dob', 'incl_excl_grp']].apply(lambda x: x.ffill().bfill()).reset_index(level=0, drop=True)
        data_df['demo_sex'] = data_df['demo_sex'].replace([0, 1], ['F', 'M'])
        data_df['visit_date'] = data_df['visit_date'].fillna(data_df['mo3fupc_date']) # 3 month visit doesn't have the usual visit_date col
    else:
        # LoTS
        data_df = data_df.set_index(['demo_study_id', 'redcap_event_name'])
        data_df[['demo_sex', 'demo_dob']] = data_df[['demo_sex', 'demo_dob']].apply(lambda x: x.ffill().bfill())
        data_df['demo_sex'] = data_df['demo_sex'].replace([1, 2], ['F', 'M'])

    # data_df.to_csv('data_df_before_visit_date_fix.csv')
    data_df['visit_date'] = pd.to_datetime(data_df['visit_date'], format='mixed')
    data_df['interview_age'] = (data_df['visit_date'] - pd.to_datetime(data_df['demo_dob'])).apply(lambda x: round(.0328767*x.days) if pd.notnull(x) else np.nan)

    if fields_to_withhold:
        drop_cols = [ col for pattern in fields_to_withhold for col in data_df.columns if re.match(pattern, col) ]

    if item_level_replacements:
        replace_df = pd.read_csv(item_level_replacements)
    else:
        replace_df = None

    # drop broken COVID checkbox question
    data_df = data_df.drop(columns=['cov2y_40___1', 'cov2y_40___2', 'cov2y_40___3', 'cov2y_40___4', 'cov2y_40___5', 'cov2y_40___6', 'cov2y_40___7', 'cov2y_40___8', 'cov2y_40___exercised'], errors='ignore')
    # print('DATA_DICT_DF COLUMNS = {}'.format(data_dict_df.columns))
    # data_dict_df = data_dict_df[data_dict_df]='', errors='ignore')

    # convert all checkboxes to labels
    checkbox_fields = data_dict_df[data_dict_df['type'] == 'checkbox'].index
    for field in checkbox_fields:
        choices_str = data_dict_df.xs(field)['choices']
        use_label = (field in CHECKBOX_TO_LABEL) # check if combined response should be values or labels
        if not field + '___1' in data_df.columns:
            continue
        data_df =  replace_checkbox_with_label(data_df, field, choices_str, use_label)

    # replace study staff/doctor names with generic references
    data_df = replace_staff_names(data_df)

    # fields required by NIH forms that don't appear in matched REDCap form
    if 'incl_excl_grp' in data_df.columns:
        form_field_map = {
            'ndar_subject01': ['incl_excl_grp', 'demo_race'],
            'srs02': ['incl_excl_grp', 'demo_completed_by'],
            'cbcl1_501': ['incl_excl_grp'],
            'socdemo01': ['cbcl_grade_in_school']
        }
    else:
        form_field_map = {
            'ndar_subject01': ['expert_diagnosis_tourette','expert_diagnosis_chronic_tics','expert_diagnosis_transient','demo_race'],
            'srs02': ['expert_diagnosis_tourette','expert_diagnosis_chronic_tics','expert_diagnosis_transient','demo_completed_by'],
            'socdemo01': ['cbcl_grade_in_school']
        }

    # data_df.to_csv('data_df_before_form_loop.csv')

    for form in nih_forms:
        required_fields = [ 'subjectkey', 'visit_date', 'interview_age', 'demo_sex'] # fields shared by every NIH form

        upload_file = os.path.join(output_directory, '{}.csv'.format(form))
        if not redo and os.path.exists(upload_file):
            continue

        # write NIH header for submission file (nih form name, nih form version)
        with open(upload_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(re.match(r'(\w+)(\d{2}$)', form).groups()))

        redcap_forms = list(form_map_df.xs(form).values.flatten()) # get all REDCap forms associated with current NIH form
        print('NIH_FORM = {}, REDCAP_FORMS = {}'.format(form, redcap_forms))

        # Get columns in REDCap forms
        form_dd_df = data_dict_df[data_dict_df['form'].isin(redcap_forms)]
        form_cols = list(form_dd_df[form_dd_df['type'] != 'descriptive'].index.values)

        # keep form-specific cols, shared cols, and form-specific required cols (that exist outside current forms)
        keep_cols = form_cols + required_fields + ['demo_study_id', 'redcap_event_name']
        if form in form_field_map:
            keep_cols += form_field_map[form]
        form_df = data_df.reset_index()[np.unique(keep_cols)].drop(columns='index', errors='ignore')
        # form_df.to_csv(os.path.join(output_directory,'form_df_after_keep_cols.csv'))

        # remove empty rows
        subset = [ col for col in form_cols if col not in fields_to_withhold + required_fields ]
        if form != 'tsp01': # since we're joining to a different csv for tsp + most subjects do not have new form data, we cannot drop here
            form_df = form_df.dropna(how='all', subset=subset)

        form_df = form_df.rename(columns={'demo_sex': 'sex'})

        # convert numbers to label where needed
        convert_cols = [ col for pattern in OTHER_LABELS_NEEDED for col in form_df.columns if re.match(pattern, col) ]
        for col in convert_cols:
            choices_str = data_dict_df.xs(col)['choices']
            # print('col = {}, choices_str = {}'.format(col,choices_str))
            form_df =  replace_num_with_label(form_df, col, choices_str)

        ### handle event name column from REDCap export
        # NOTE: This dict contains both NewTics R01 and LoTS 2.0 event names
        event_name_mapping = \
            {'screening_visit_arm_1': 'Screening', \
            'initial_scan_visit_arm_1': 'Initial Scan', \
            'repeat_scan_visit_arm_1': 'Repeat Scan', \
            '3_month_follow_up_arm_1': '3 Month Follow-up', \
            '12_month_follow_up_arm_1': '12 Month Follow-up', \
            'clinical_follow_up_arm_1': 'Clinical Follow-up 1', \
            'clinical_follow_up_arm_1b': 'Clinical Follow-up 2', \
            'clinical_follow_up_arm_1c': 'Clinical Follow-up 3', \
            'clinical_follow_up_arm_1d': 'Clinical Follow-up 4', \
            'clinical_visit_1_arm_1': 'Clinical Visit 1', \
            'scan_visit_1_arm_1': 'Scan Visit 1', \
            'repeat_scan_visit_arm_1': 'Repeat Scan Visit 1', \
            '3_month_survey_arm_1': '3 Month Survey', \
            '6_month_survey_arm_1': '6 Month Survey', \
            '9_month_survey_arm_1': '9 Month Survey', \
            '12_month_survey_arm_1': '12 Month Survey', \
            '15_month_survey_arm_1': '15 Month Survey', \
            '18_month_survey_arm_1': '18 Month Survey', \
            '21_month_survey_arm_1': '21 Month Survey', \
            '24_month_clinical_arm_1': '24 Month Clinical Visit', \
            '24_month_scan_visi_arm_1': '24 Month Scan Visit', \
            '24_mo_survey_delay_arm_1': '24 Mo Survey Delayed 24 Mo', \
            '27_mo_survey_delay_arm_1': '24 Mo Survey Delayed 27 Mo', \
            '30_mo_survey_delay_arm_1': '24 Mo Survey Delayed 30 Mo', \
            '33_mo_survey_delay_arm_1': '24 Mo Survey Delayed 33 Mo', \
            'wk_1_day_1_arm_1': 'Week 1 Day 1', \
            'wk_1_day_2_arm_1': 'Week 1 Day 2', \
            'wk_1_day_3_arm_1': 'Week 1 Day 3', \
            'wk_1_day_4_arm_1': 'Week 1 Day 4', \
            'wk_1_day_5_arm_1': 'Week 1 Day 5', \
            'wk_1_day_6_arm_1': 'Week 1 Day 6', \
            'wk_1_day_7_arm_1': 'Week 1 Day 7', \
            'wk_2_day_1_arm_1': 'Week 2 Day 1', \
            'wk_2_day_2_arm_1': 'Week 2 Day 2', \
            'wk_2_day_3_arm_1': 'Week 2 Day 3', \
            'wk_2_day_4_arm_1': 'Week 2 Day 4', \
            'wk_2_day_5_arm_1': 'Week 2 Day 5', \
            'wk_2_day_6_arm_1': 'Week 2 Day 6', \
            'wk_2_day_7_arm_1': 'Week 2 Day 7'}
            
        event_name_renames_tsp = \
            ['Screening', \
            '3 Month Follow-up', \
            '12 Month Follow-up', \
            'Clinical Follow-up 1', \
            'Clinical Follow-up 2', \
            'Clinical Follow-up 3', \
            'Clinical Follow-up 4']
        
        # form_df.to_csv(os.path.join(output_directory,'form_df_before_event_rename.csv'))
        form_df['redcap_event_name'] = form_df['redcap_event_name'].replace(to_replace=event_name_mapping)
        rename = 'visit' # rename "redcap_event_name" depending on form
        form_df = form_df.rename(columns={'redcap_event_name': rename})

        # print(form_df.columns)
        if to_date:
            form_df = form_df[form_df['visit_date'] < to_date] # remove rows newer than date

        # Re-index by demo_study_id and visit
        form_df.set_index(['demo_study_id','visit'])

        # remove test subjects
        form_df = form_df[~form_df.demo_study_id.str.contains('test')]

        # ndar_subject01
        #   set our dna sample type to saliva, change value of sample usability to string, set required variables about type of study,
        #   specify required phenotype description (taken from iec01 group requirement), set race to 'More than one race' if they checked
        #   multiple
        if form == 'ndar_subject01':
            if 'incl_excl_grp' in form_df.columns:
                form_df['sample_description'] = np.where(form_df['dna_sample_collected'] == 1, 'saliva', np.nan)
                form_df['dna_sample_usable'] = form_df['dna_sample_usable'].replace([1,2], ['Sample usable', 'Sample not usable'])
                form_df = form_df.assign(twins_study = 'No', sibling_study = 'No', family_study = 'No')
                conditions = [(form_df['incl_excl_grp'] == 1), (form_df['incl_excl_grp'] == 2), (form_df['incl_excl_grp'] == 3)]
                choices = ['Tics now, but developed them only in the past 6 months', 'Meets DSM-5 criteria for Tourettes', 'No history of tics']
                form_df['phenotype_description'] = np.select(conditions, choices, default='999')
                form_df['demo_race'] = form_df['demo_race'].where(~form_df['demo_race'].str.contains(';'), 'More than one race')
            else:
                pass

        # phenotype required
        #   some forms (i.e. srs02, ndar_subject01) require phenotype - for all such forms (except iec01 that's already mapped as numbers)
        #   we want to change the field name and use the concatenated labels
        if form != 'iec01' and 'incl_excl_grp' in keep_cols:
            choices_str = data_dict_df.xs('incl_excl_grp')['choices']
            form_df =  replace_num_with_label(form_df, 'incl_excl_grp', choices_str).rename(columns={'incl_excl_grp': 'phenotype'})

        # socdemo01
            # only report one race (store others in separate var), add 'years' to education string, swap hispanic codes, make sure education
            #   is within string limits
        if form == 'socdem01':
            form_df = form_df.apply(separate_multi_response, args=('demo_race', 'fsprg'), axis=1)
            form_df['demo_ethnicity'] = form_df['demo_ethnicity'].replace(['1', '2'], ['2', '1'])
            form_df['demo_race'] = form_df['demo_race'].replace('Native Hawaiian or Other Pacific Islander', 'Hawaiian or Pacific Islander')
            form_df['demo_mat_edu'] = form_df['demo_mat_edu'].apply(add_years_str).str.slice(0,20)
            form_df['demo_pat_edu'] = form_df['demo_pat_edu'].apply(add_years_str).str.slice(0,20)

         # bisbas01
            # replace field names
        if form == 'bisbas01':
            # Rename only specified bisbas_ columns to bb_
            bisbas_cols_to_rename = [
                'bisbas_child_scr_bas_dr',
                'bisbas_child_scr_bas_dr_mean',
                'bisbas_child_scr_bas_fs',
                'bisbas_child_scr_bas_fs_mean',
                'bisbas_child_scr_bas_rr',
                'bisbas_child_scr_bas_rr_mean',
                'bisbas_child_scr_bis',
                'bisbas_child_scr_bis_mean',
                'bisbas_prnt_scr_bas_drive',
                'bisbas_prnt_scr_bas_drive_mean',
                'bisbas_prnt_scr_bas_reward',
                'bisbas_prnt_scr_bas_reward_mean',
                'bisbas_prnt_scr_bis_mean',
                'bisbas_prnt_scr_bis_raw',
                'bisbas_prnt_scr_fun',
                'bisbas_prnt_scr_fun_mean'
            ]
            rename_dict = {col: col.replace('bisbas_', 'bb_') for col in bisbas_cols_to_rename if col in form_df.columns}
            if 'demo_study_id' in form_df.columns:
                rename_dict['demo_study_id'] = 'src_subject_id'
            form_df = form_df.rename(columns=rename_dict)

            # Add version_form column based on bis_bas_child_1 value
            if 'bis_bas_child_1' in form_df.columns:
                form_df['version_form'] = form_df['bis_bas_child_1'].apply(lambda x: 'child form' if pd.notnull(x) and isinstance(x, (int, float)) and x > 0 else 'parent form')

        # cash_choice01
        #   replace field names
        if form == 'cash_choice01':
            rename_dict = {}
            if 'demo_study_id' in form_df.columns:
                rename_dict['demo_study_id'] = 'src_subject_id'
            form_df = form_df.rename(columns=rename_dict)
        # bsmss01 (ses)
        #   only report partner info for 'primary residence' parent's partner, remove occ details because of PHI issues
        if form == 'bsmss01':
            form_df = form_df.assign(version_form = np.nan, bsmss03_spouse = np.nan, bsmss07_spouse = np.nan)
            if 'primary_residence' in form_df.columns:
                # print('######## replacing primary_residence label #########')
                form_df = form_df.rename(columns={'primary_residence': 'ses_primary_res'})
            # form_df.to_csv('form_df_before_ses_fix.csv')
            form_df = form_df.apply(ses_primary_partner, axis=1)

            # Make some columns ints
            # form_df['bsmss03_spouse'] = form_df['bsmss03_spouse'].astype('float64', errors='ignore')
            # form_df['bsmss07_spouse'] = form_df['bsmss07_spouse'].astype('float64', errors='ignore')
            form_df['bsmss03_spouse'] = form_df['bsmss03_spouse'].astype('str').apply(sparse_int_fix)
            form_df['bsmss07_spouse'] = form_df['bsmss07_spouse'].astype('str').apply(sparse_int_fix)

            drop_cols += ['ses_primary_res', 'ses_detail_occ_mother', 'ses_detail_occ_father'] + [ col for col in form_df.columns if col.endswith('partner') ]

        # mab01 (maternal/birth history)
        #   remove 'unknown' code for apgar responses, convert the task ages to months, make gest_wks is an integer
        if form == 'mab01':
            form_df['matern_no_preg'] = form_df['matern_no_preg'].replace('unknown', np.nan).astype(float) # 'unknown' strings break output type -- convert to float after correction
            form_df['matern_no_preg'] = form_df['matern_no_preg'].astype('int', errors='ignore')
            form_df['matern_no_births'] = form_df['matern_no_births'].astype('int', errors='ignore')
            form_df[['matern_gest_wks']] = form_df[['matern_gest_wks']].apply(age_to_units, args=(AgeUnits.WEEKS,), axis=1)
            form_df[['matern_nicu_days']] = form_df[['matern_nicu_days']].apply(age_to_units, args=(AgeUnits.DAYS,), axis=1)
            apgar_cols = [ col for col in form_df.columns if col.startswith('matern_apgar') ]
            form_df[apgar_cols] = form_df[apgar_cols].replace(11, np.nan)
            matern_age_tsks = [ 'matern_' + col for col in ['hld_head', 'rll_ovr', 'toy_reach', 'sat_up', 'fing_fed', 'crawl', 'pull_to_stand', 'walk', 'slf_fed', 'tlk_wrd_combo'] ]
            form_df[matern_age_tsks] = form_df[matern_age_tsks].apply(age_to_units, args=(AgeUnits.MONTHS,), axis=1)

        # tichist01 (family history)
        #   remove relative_ from fh columns to fit within character limit
        if form == 'tichist01':
            relative_renames = { col: col.replace('relative_', '') for col in form_df.columns if re.match(r'fh_\w+_other_relative_relationship_800', col) }
            form_df =  form_df.rename(relative_renames, axis=1)

        # srs02
        #   map question responses to actual values (not scores), create separate sex norm columns for scores and rename, add columns
        #   for who completed form (taken from demo_completed_by field)
        if form == 'srs02':
            srsq_cols = get_matching_cols(form_df, r'srs_800_q\d+')
            form_df[srsq_cols] = form_df[srsq_cols].apply(lambda x: x.str[0], axis=1)
            raw_score_cols = [ 'srs_awareness', 'srs_cognition', 'srs_communication', 'srs_motivation', 'srs_mannerisms', 'srs_total' ]
            tscore_cols = [ col + '_t' for col in raw_score_cols ]
            form_df =  sex_norm_srs(form_df, raw_score_cols, 'raw')
            form_df =  sex_norm_srs(form_df, tscore_cols, 'tscore')
            form_df['srs_score'] = np.nan
            form_df['respond_detail'] = form_df['demo_completed_by'].apply(mom_or_dad)
            form_df['respond'] = form_df['demo_completed_by'].apply(respond_code)
            for col in ['respond', 'respond_detail']:
                form_df[col] = form_df[col].replace(1.0,'1')
                form_df[col] = form_df[col].replace(2.0,'2')
                form_df[col] = form_df[col].replace(3.0,'3')
                form_df[col] = form_df[col].replace(4.0,'4')
                form_df[col] = form_df[col].replace(5.0,'5')
                form_df[col] = form_df[col].replace(6.0,'6')
                form_df[col] = form_df[col].replace(7.0,'7')
                form_df[col] = form_df[col].replace(8.0,'8')
                form_df[col] = form_df[col].replace(9.0,'9')
                form_df[col] = form_df[col].replace(999.0,'999')
            form_df = form_df.drop(columns='demo_completed_by')

        # ticscreener01
        #   only report most specific value of symptom checklist timeframe (i.e. 'past week' if both 'past week' and 'lifetime' selected),
        #   ensure consistent values between different visit forms (i.e. make 'past week' always be 2, introduce new code for 'since last visit')
        if form == 'ticscreener01':
            if 'exp_specify_tics_2_pw' in form_df.columns:
                form_df = form_df.drop(columns=['exp_specify_tics_2_pw'], errors='ignore')
                form_df = split_multiform_row(form_dd_df, nih_dd_directory, form, form_df, update_ticscreener)
            else:
                pass

        # pedsql01
        #   replace version numeric code with string representing version, decrement question scale to 0-4
        if form == 'pedsql01':
            q_cols = get_matching_cols(form_df, r'pedsql_\w*_\d')
            form_df[q_cols] = form_df[q_cols] - 1
            form_df['pedsql_version'] = form_df['pedsql_version'].apply(lambda x: 'Version for Ages ' + x if pd.notnull(x) else np.nan)
            form_df = form_df.rename(columns={"pedsql_version": "version_form"})

        # adhdrs01
        #   combine all ADHD rating scale forms into one and use column version form to differentiate lifetime/expert/parent
        if form == 'adhdrs01':
            form_df = split_multiform_row(form_dd_df, nih_dd_directory, form, form_df, update_adhdrs)

        # mvhsp01
        #   recode completed_by, and record visit type under version form
        if form == 'mvhsp01':
            if 'med_completed_by_12mo' in form_df.columns:
                replace_cols = ['med_completed_by', 'med_completed_by_12mo']
                form_df[replace_cols] = form_df[replace_cols].replace(range(1,5), 8)
                form_df[replace_cols] = form_df[replace_cols].replace([5, 6], [17, 98])
                form_df = split_multiform_row(form_dd_df, nih_dd_directory, form, form_df, None)
                # print(form_df.columns)
                form_df['chf_05'] = form_df['chf_05'].str.slice(0,250)
            else:
                form_df.drop(columns=['med_completed_by'], errors='ignore')

        # puts01
        #   specify that we used 9-item PUTS
        if form == 'puts01':
            form_df['version_form'] = '9-item PUTS'

        # pds01
        #   add column 'respond', all values of 999 (respondent not specified)
        if form == 'pds01':
            form_df['respond'] = 999

        # ygtss01
        #   remove '_past_week' to meet column length limits
        if form == 'ygtss01':
            # rename some columns
            form_df =  form_df.rename(columns={'ygtss_motor_tic_score':'ygtss_motor_total', 'ygtss_phonic_tic_score': 'ygtss_phonic_total'})
            long_cols = ['ygtss_past_week_unavailable_list', 'ygtss_past_week_expert_total_tic_score']
            form_df =  form_df.rename({ col: col.replace('past_week_', '') for col in form_df.columns if col in long_cols }, axis=1)
            form_df = split_multiform_row(form_dd_df, nih_dd_directory, form, form_df, update_ygtss)


        # erd_tics01
        #   rename impairment col to meet char limit, remove code for blank, reverse-code ocd response
        if form == 'erd_tics01':
            form_df = form_df.rename({ 'expert_diagnosis_impairment_distress': 'expert_impairment_distress'}, axis=1)
            form_df.replace(4, np.nan, inplace=True)
            form_df['expert_diagnosis_ocd'].replace([1,2,3], [3,2,1], inplace=True)
            form_df['expert_diagnosis_onset_notes'] = form_df['expert_diagnosis_onset_notes'].str.slice(0,20)
            # remove columns that are not in the NDAR data dictionary
            erd_tics01_drop_cols = [
                'expert_diagnosis_dsto', 
                'expert_diagnosis_rater',
                'expert_diagnosis_tic_age_2', 
                'expert_diagnosis_tic_onset', 
                'expert_ts_dx_dsmiv', 
                'expert_ts_dx_dsmiv_tr', 
                'expert_ts_dx_dsmv']
            form_df = form_df.drop(columns=erd_tics01_drop_cols, errors='ignore')

        # cybocs01
        #   label visit/form type (i.e. worst ever, past week), recode 'past week' to always be 3, add unique code for 'since last visit'
        if form == 'cybocs01':
            form_df = form_df.drop(columns='comp_counting_12mo', errors='ignore') # FIXME: column is meant to be descriptive; remove once we have updated data dict
            form_df = split_multiform_row(form_dd_df, nih_dd_directory, form, form_df, update_cybocs)
            for col in [ col for col in form_df if 'spec' in col ]:
                form_df[col] = form_df[col].apply(lambda x: x[:100] if pd.notnull(x) else x)
            form_df['ocd_aware'] = form_df['ocd_aware'].replace('2;3', '2') # question is when does child *first* become aware; 'at start' (2) comes before 'during' (3)

        # tic_outcome_data01
        #   recode no=0, yes=1 for consistency
        if form == 'tic_outcome_data01':
            form_df = form_df.replace([1,2], [0,1])

        # ksads_dsm5_diagnoses
        #   recode no=0, yes=1, na = 88
        if form == 'ksads_dsm5_diagnoses':
            form_df[['ksads_outpt', 'ksads_psyhosp']] = form_df[['ksads_outpt', 'ksads_psyhosp']].replace([1,2,3], [88,0,1])
            form_df['expert_diagnosis_onset_notes'] = form_df['expert_diagnosis_onset_notes'].str.slice(0,20)

        # pccf01
        #   remove double-mapped date
        if form == 'pccf01':
            drop_cols.append('mo3fupc_date')

        if form == 'paness01':
            drop_cols += ['pan_inc_desc', 'pan_oth_descr']
            form_df['pan_chorea'] = form_df['pan_chorea'].replace([2,3], 1) # convert to yes/no

        # cbcl01 + cbcl1_501
        #   Remove text fields from CBCL questionnaires (some fields like groups contain pseudo-PHI; and they're not really necessary)
        if 'cbcl' in form:
            redcap_form = 'child_behavior_checklist_cbcl_ages_6_18' if form == 'cbcl01' else 'ycbcl_age_5'
            qcols = list(form_dd_df[form_dd_df['form'] == redcap_form].index.values)
            qtext_fields = [ col for col in form_df if col in qcols and col not in list(form_dd_df[form_dd_df['type'] == 'radio'].index.values) ]
            form_df = form_df.drop(columns=qtext_fields)

        # cbcl01
        #   remove repeated demographic info, decrement certain scales to begin at 0, consolidate 'Never' + 'Less than 1' into same answer,
        #   set required cols from newer CBCL version to unkwown (999)
        if form == 'cbcl01':
            # check for NewTics or LoTS
            if 'cbcl_1' in form_df.columns:
                # get cbcl columns in order they appear in form, so we can use column ranges
                ordered_cbcl_columns = [ x for x in data_dict_df[data_dict_df['form'] == 'child_behavior_checklist_cbcl_ages_6_18'].index.tolist() if x in form_df.columns ]
                ordered_cbcl_columns = [ col for col in form_df.columns if col not in ordered_cbcl_columns ] + ordered_cbcl_columns
                form_df = form_df[ordered_cbcl_columns]
                form_df.loc[:, 'cbcl_1':'cbcl_113c']= form_df.loc[:, 'cbcl_1':'cbcl_113c'].replace([1,2,3], [0,1,2])
                form_df.loc[:, 'cbcl_number_of_sports':'cbcl_chores_well3'] = form_df.loc[:, 'cbcl_number_of_sports':'cbcl_chores_well3'].replace([1,2,3,4,5], [0,1,2,3,999])
                form_df.loc[:, 'cbcl_hobbies1_a':'cbcl_hobbies3_b'] = form_df.loc[:, 'cbcl_hobbies1_a':'cbcl_hobbies3_b'].replace(999, 9) # hobbies/activities has different NA code
                form_df.loc[:, 'cbcl_close_friends':'cbcl_disability'] =  form_df.loc[:, 'cbcl_close_friends':'cbcl_disability'].replace([1,2,3,4,5], [0,1,2,3,9])
                form_df['cbcl_times_a_week_friends'] = form_df['cbcl_times_a_week_friends'].replace([1,2,3], [0,1,2]) # after decrementing range, go back and adjust this specific col
                fill_cols = ['cbcl_9', 'cbcl_28', 'cbcl_33', 'cbcl_41', 'cbcl_49', 'cbcl_56d', 'cbcl_56h', 'cbcl_59', 'cbcl_76', 'cbcl_82', 'cbcl_84', 'cbcl_85', 'cbcl_98', 'cbcl_106', 'cbcl_110', 'cbcl_112' ] + [ col for col in form_df.columns if re.match('cbcl_(raw|t|per)_(activities|social|school|totalcomp)', col) ]
                form_df[fill_cols] = form_df[fill_cols].fillna(999)
                for col in fill_cols:
                    form_df[col] = form_df[col].astype('int')
                form_df.loc[:, 'cbcl_sports1_a':'cbcl_sports3_b'] = form_df.loc[:, 'cbcl_sports1_a':'cbcl_sports3_b'].fillna(999) # sports only is marked as required...

                missing_cols = [ 'cbcl_' + col for col in ['emotional', 'sleep', 'pervasive', 'depresspr'] ]
                missing_cols = list(chain(*[ [col, col + '_raw'] for col in missing_cols ])) # need cols with and without '_raw' suffix
                for col in missing_cols:
                    form_df[col] = 999

                drop_cols += [ 'cbcl_' + col for col in ['age', 'date', 'sex', 'gender', 'race', 'ethnicity', 'grade_in_school', 'attending_school',
                    'informant_gender', 'informant_relation', 'notes']] # dropping notes as cbcl_comments maps to same field and cbcl_notes is unused

                # drop rows where the ASEBA scoring was not done
                form_df.dropna(subset=['cbcl_age_6_18_yn'], inplace=True)
            else:
                pass

        # cbcl1_501
        #   Recode 3 as 4 (NA), set unrecorded newer columns to unknown (999), cap concerns response
        if form == 'cbcl1_501':
            form_df['ycbcl_age_5_yn'] = form_df['ycbcl_age_5_yn'].replace(3, 4)
            for col in ['cbcl_depresspr', 'cbcl_depresspr_raw']:
                form_df[col] = 999
            fill_cols = ['ycbcl_q11']
            form_df[fill_cols] = form_df[fill_cols].fillna(999)
            int_cols = ['ycbcl_q74', 'ycbcl_q76', 'ycbcl_q86', 'ycbcl_q92']
            for col in int_cols:
                form_df[col] = form_df[col].replace(0.0,'0')
                form_df[col] = form_df[col].replace(1.0,'1')
                form_df[col] = form_df[col].replace(2.0,'2')
            drop_cols += [ 'cbcl_gender' ]
            form_df = form_df[form_df.ycbcl_age_5_yn.notnull()]

        # cptc01
        #   Rename cols to fit within col name limits
        if form == 'cptc01':
            rename_cols = [ col for col in form_df.columns if re.match(r'cpt_hit\w*_change_guideline$', col) ]
            form_df = form_df.rename(columns={ col: col.replace('cpt_', '') for col in rename_cols })

            form_df['cpt_type'] = np.where(form_df['interview_age'] >= 72, 1, 2) # set assessment to KCPT if age < 6

            guideline_cols = [ col for col in form_df.columns if col.endswith('guideline') ]
            form_df[guideline_cols] = form_df[guideline_cols].apply(lambda x: x.str.lower())
            guideline_cols.remove('cpt_hit_rt_guideline')
            form_df[guideline_cols] = form_df[guideline_cols].replace('within average range', 'within the average range')
            # Clean up cpt_commissions_pctile and cpt_detectability_pctile: convert '65th' -> 65
            for col in ['cpt_commissions_pctile', 'cpt_detectability_pctile', 'cpt_hit_rt_pctile', 'cpt_omissions_pctile', 'cpt_hit_rt_block_change_pctile', 'cpt_hit_rt_isi_change_pctile', 'cpt_hit_rt_std_error_pctile', 'cpt_perseverations_pctile', 'cpt_variability_pctile', 'cpt_detectability_t']:
                if col in form_df.columns:
                    form_df[col] = form_df[col].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64')

            # For cpt_commissions_n and cpt_omissions_n, extract number before %
            for col in ['cpt_commissions_n', 'cpt_omissions_n', 'cpt_perseverations_n']:
                if col in form_df.columns:
                    form_df[col] = form_df[col].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64')


        # sldc01
        #   Split adhd comb/inatt/hyper fields into separate upload files
        if form == 'sldc01':
            for idx, type in enumerate(['comb', 'inatt', 'hyper']):
                adhd_type_cols = [ col for col in form_df.columns if col.startswith('ksads5_adhd_' + type) ]
                adhd_type_df = form_df[adhd_type_cols + [ col for col in form_df.columns if not col.startswith('ksads5') ]]
                adhd_type_df.assign(sldc153=idx, sldc153c=idx)
                adhd_type_df['visit_date'] = format_date_str(adhd_type_df['visit_date'])
                adhd_type_file = os.path.join(output_directory, 'sldc01_adhd_{}.csv'.format(type))
                with open(adhd_type_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(list(re.match(r'(\w+)(\d{2}$)', form).groups()))
                adhd_type_df.to_csv(adhd_type_file, mode='a', index=False, float_format='%g')
                form_df = form_df.drop(columns=adhd_type_cols)

        # tsp01
        #   Combine with drz_output for summary metrics, remap drz_tics to comments for extra length (it's more often used as a general
        #   comment field anyways), drop completely empty rows after merge
        if form == 'tsp01':
            drz_score_cols = ['demo_study_id', 'version_form', 'session', 'condition', 'tic_freq', 'tsp_tfi', 'duration', 'data_file1', 'notes']
            drz_score_df = pd.read_csv(os.path.join(BASE_PATH, 'TSP', 'drz_output_with_trainer.csv'), skiprows=0, names=drz_score_cols)
            # print(drz_score_df)
            drz_score_df['version_form'] = drz_score_df['version_form'].replace(['screen'] + [ str(d) + 'mo' for d in [3,12,24,36,48,60] ], event_name_renames_tsp)
            drz_score_df['visit'] = drz_score_df['version_form']

            form_df = form_df.merge(drz_score_df, on=['demo_study_id', 'visit'], how='inner')
            form_df['int_dur'] = 10 # duration of tic-free intervals
            form_df['version_form'] = form_df[['version_form', 'condition', 'session']].apply(lambda x: '; '.join(x.astype(str)), axis=1)
            form_df['data_file1'] = form_df['data_file1'].astype(str).apply(lambda x: os.path.basename(x)) # filename only since on same path as submission forms and path contains userid
            form_df = form_df.drop(columns=['condition', 'session', 'duration', 'notes', 'visit'])
            form_df = form_df.rename(columns={'drz_tics': 'comments'})

            form_df = form_df.dropna(how='all', subset=[col for col in subset + ['tic_freq', 'tsp_tfi', 'data_file1'] if col in form_df])
            tsp01_drop_cols = ['drz_tics_knwn_prev']
            form_df = form_df.drop(columns=tsp01_drop_cols, errors='ignore')

        # endvisit01
        #   Remove "3" (legal guardian) until we ask NDAR to add it as an option for visit_parents_present
        if form == 'endvisit01':
            # replace 3 with blank
            form_df['visit_parents_present'] = form_df['visit_parents_present'].replace('3','')
            # replace 1;2 with 1
            form_df['visit_parents_present'] = form_df['visit_parents_present'].replace('1;2','1')
            # drop some columns not in NDAR data dictionary
            endvisit01_drop_cols = [
                'visit_covid19_mod', 
                'visit_tics_father', 
                'visit_tics_mother', 
                'yrs_w_tics_at_visit']
            form_df = form_df.drop(columns=endvisit01_drop_cols, errors='ignore')


        if replace_df:
            # replace specific items that are known to be problematic / missing (documented in cfg/item_level_replacements spreadsheet)
            form_replace_df = replace_df[replace_df['form'] == form]
            form_replace_df['visit_date'] = pd.to_datetime(form_replace_df['visit_date'])
            for _, row in form_replace_df.iterrows():
                form_df.loc[
                    (form_df['demo_study_id'] == row['demo_study_id']) & (form_df['visit_date'] == row['visit_date']),
                    row['variable']
                ] = row['new_value']

        # convert all text-field ages to months
        age_to_months_cols = [ col for col in form_df.columns if re.match(r'(matern|ksads|ksads5|fh)(_\w*)*_age', col) and \
            col not in list(form_dd_df[form_dd_df['type'] == 'radio'].index.values) ]
        if age_to_months_cols:
            # assume ages (other than ones in mab01 which ask for intended unit) will be reported in years
            form_df[age_to_months_cols] = form_df[age_to_months_cols].apply(age_to_units, args=(AgeUnits.MONTHS, AgeUnits.YEARS), axis=1)

        if form not in 'socdem01':
            form_df = form_df[form_df['visit_date'] > datetime(2017, 8, 1)] # remove pre-R01 rows

        form_df['visit_date'] = format_date_str(form_df['visit_date']) # (after all date comparisons) revert visit date to required string format

        # write out data to separate NIH form files
        form_df = form_df.drop(columns=drop_cols + [col for col in form_df.columns if col.endswith('_complete')], errors='ignore')
        if form == 'kbit_ii01':
            kbit_field_renames = {
                'kbit_matrices_raw': 'kbit_nonverbal_score', 
                'kbit_nonverbal_standard': 'kbit_nonverbalstand_score', 
                'kbit_riddles_raw': 'kbit_riddle_score', 
                'kbit_verbal_knowledge_raw': 'kbit_verb_score', 
                'kbit_verbal_plus_nonverbal': 'kbit_iqsum_score', 
                'kbit_verbal_raw': 'kbit_verbalsum_score', 
                'kbit_verbal_standard': 'kbit_verbalstand_score', 
                'kbit_iq': 'kbit_iqstand_score'}
            form_df = form_df.rename(columns=kbit_field_renames)
            int_cols = ['kbit_iqstand_score','kbit_nonverbal_score','kbit_nonverbalstand_score','kbit_riddle_score','kbit_verb_score', 
                'kbit_iqsum_score','kbit_verbalsum_score','kbit_verbalstand_score']
            for col in int_cols:
                form_df[col] = form_df[col].astype('float64', errors='ignore')
            form_df.to_csv(upload_file, mode='a', index=False, float_format='%d')
        elif form == 'cbcl01':
            if 'cbcl_1' in form_df.columns:
                int_cols = ['cbcl_25', 'cbcl_72', 'cbcl_83', 'cbcl_84', 'cbcl_85', 'cbcl_90', 'cbcl_105']
                for col in int_cols:
                    form_df[col] = form_df[col].astype('float64', errors='ignore')
                form_df.to_csv(upload_file, mode='a', index=False, float_format='%d')
            else:
                pass
        elif form == 'mab01':
            int_cols = ['matern_no_preg', 'matern_no_births']
            for col in int_cols:
                form_df[col] = form_df[col].astype('float64', errors='ignore')
            form_df.to_csv(upload_file, mode='a', index=False, float_format='%d')
        elif form == 'bsmss01':
            # print('########### bsmss01 to_csv #############')
            # print(form_df.dtypes)
            form_df.to_csv(upload_file, mode='a', index=False, float_format='%d')
        else:
            form_df.to_csv(upload_file, mode='a', index=False, float_format='%g')

    return
