import csv
import numpy as np
import pandas as pd
import re
import win32com.client
import sys

from datetime import datetime
from enum import Enum
from getpass import getuser, getpass
from gooey import Gooey, GooeyParser
from itertools import chain
from os.path import exists, join

sys.path.append(r'C:\Users\{}\Documents\NewTics\newtics_redcap'.format(getuser()))
from score_redcap_data import get_redcap_project, merge_projects

BASE_PATH = r'C:\Users\{}\Box\Black_lab\projects\TS\New Tics R01\Data'.format(getuser())


GUID_PATH = join(BASE_PATH, r'NIMH GUID\GUIDs.xlsx')

WITHHOLD = [ 'incl_excl_ic', 'incl_excl_who', 'incl_excl_new_tics_grp', 'share_data_permission', 'share_data_comments',
    'r01_survey_consent', 'demo_dob', 'childs_age', 'incl_excl_fon_scrn', 'dna_sample_lab_id', 'cbcl_birthdate', 'mo3fupc_who',
    '\w*_data_files*$', 'age_at_visit', 'visit_referral_source' ]
CHECKBOX_TO_LABEL = [ 'incl_excl_concom_meds', 'demo_race']
OTHER_LABELS_NEEDED = [ 'demo_\w*_mari', 'srs_800_q\d+', 'pedsql_version', 'ksads5_asd_specify' ]

NUMBER_PATTERN = '(\d+(?:\.\d+)?)'

class AgeUnits(Enum):
    YEARS = 0
    MONTHS = 1
    WEEKS = 2
    DAYS = 3


def get_matching_cols(df, pattern):
    return [ col for col in df.columns if re.match(pattern, col) ]


def get_data_dict_options_map(df, data_dict_df, variable):
    choices_str = data_dict_df.xs(variable)['choices']
    options = [ choice.split(',', 1) for choice in choices_str.split('|') ]
    options = [ opt for opt in options if opt != ['']] # handle extra '|' at end of choice specification
    return { int(option[0].strip()): option[-1].strip() for option in options }


def concat_column_values(row, sep=';'):
    return row[row.notnull()].astype(str).str.cat(sep=sep) if pd.notnull(row).any() else np.nan


# for (all?) checkboxes
def replace_checkbox_with_label(df, data_dict_df, variable, use_label=False):
    num_to_label_map = get_data_dict_options_map(df, data_dict_df, variable)
    checkbox_cols = get_matching_cols(df, variable + '___\d+')
    for col in checkbox_cols:
        checkbox_num = re.search('___(\d+)', col).group(1)
        replace1 = num_to_label_map[int(checkbox_num)] if use_label else checkbox_num
        df[col] = df[col].replace([0, 1], [np.nan, replace1])
        df[col] = df[col].replace('None of the above', np.nan)

    df[variable] = df[checkbox_cols].apply(concat_column_values, axis=1)
    return df.drop(columns=checkbox_cols)


# for radio buttons
def replace_num_with_label(df, data_dict_df, variable):
    num_to_label_map = get_data_dict_options_map(df, data_dict_df, variable)
    df[variable] = df[variable].replace(num_to_label_map)
    return df

def age_to_units(row, dest_unit, src_unit=None):
    src_unit = dest_unit if not src_unit else src_unit
    age_searches = [ 'year|yr|y', 'month|mo|m', 'week|wk|w', 'day|d' ]
    for col in row.axes[0].tolist():
        yrs_mos_wks_days = [0, 0, 0, 0]
        no_units = re.search(NUMBER_PATTERN +'$', str(row[col]))
        if no_units:
            yrs_mos_wks_days[src_unit.value] = float(no_units.group(1))
        else:
            for i, search_str in enumerate(age_searches):
                res = re.search('(?:' + NUMBER_PATTERN + ' *(?:' + search_str +')s*)', str(row[col]), flags=re.IGNORECASE)
                yrs_mos_wks_days[i] = float(res.group(1)) if res else yrs_mos_wks_days[i]

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
    if re.match('^(mom|mother)', response, flags=re.IGNORECASE):
        return 1
    elif re.match('^(dad|father)', response, flags=re.IGNORECASE):
        return 2
    else:
        return np.nan


def ses_primary_partner(row):
    spouse_cols = [ 'ses_edu_level', 'ses_occ' ]
    mpartner_cols = [ col + '_mpartner' for col in spouse_cols ]
    fpartner_cols = [ col + '_fpartner' for col in spouse_cols ]

    if row[mpartner_cols + fpartner_cols].isnull().all():
        return row

    primary_residence = mom_or_dad(row['primary_residence'])
    if primary_residence == 1:
        row['version_form'] = 'Mother\'s partner'
        row[['bsmss03_spouse', 'bsmss07_spouse']] = row[mpartner_cols]
    elif primary_residence == 2:
        row['version_form'] = 'Father\'s partner'
        row[['bsmss03_spouse', 'bsmss07_spouse']] = row[fpartner_cols]

    return row


def add_years_str(row):
    return row + ' years' if str(row).isdigit() else row


def separate_multi_response(row, response_col, other_col=None, sep=';'):
    if pd.notnull(row[response_col]):
        responses = row[response_col].split(sep)
        row[response_col] = responses[0]
        row[other_col] = responses[1:] if (other_col and len(responses) > 1) else np.nan
    return row


def gender_norm_srs(df, cols, type):
    rename_dict = { col: type + str(i+1) for i, col in enumerate(cols[:5]) }
    total_col = 'srs_total{}'.format('_t' if type == 'tscore' else '')
    rename_dict[total_col] = '{}{}all'.format(type, 'score' if type == 'raw' else '')
    df = df.rename(columns=rename_dict)

    renamed_cols = list(rename_dict.values())
    for gender in [('M', 'male'), ('F', 'female')]:
        new_cols = [ '_'.join([gender[1], col]) for col in renamed_cols ]
        for i, col in enumerate(new_cols):
            df[col] = np.where(df['gender'] == gender[0], df[renamed_cols[i]], np.nan)
        score_all_col = [ col for col in new_cols if col.endswith('scoreall') ][0]
        df.loc[df['gender'] == gender[0], score_all_col] = df.loc[df['gender'] == gender[0], score_all_col].fillna(999)
    return df.drop(columns=renamed_cols)


def get_guid_df():
    xlApp = win32com.client.Dispatch('Excel.Application')
    xlws = xlApp.Workbooks.Open(GUID_PATH).Sheets(1)
    content = list(xlws.Range(xlws.Cells(2, 1), xlws.Cells(200,3)))
    data = [content[i:i+3] for i in range(3, len(content), 3)]
    df = pd.DataFrame(data, columns=['date', 'demo_study_id', 'subjectkey'])
    return df.set_index('demo_study_id').drop(columns='date')


def decrement_scale(df):
    return df.select_dtypes(['int']) - 1


def update_adhdrs(temp_df, form):
    who = 'expert' if 'expert' in form else 'parent'
    which = 'worst-ever' if 'lifetime' in form else 'past week'
    temp_df['version_form'] = '; '.join([who, which])
    return temp_df

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

def update_ticscreener(temp_df, form):

    if form == 'tic_symptom_checklist_screening':
        temp_df = temp_df.replace('1;2', '2') # for screening checklist, 2 (past week) is more specific than 1 (lifetime)
    else:
        temp_df = temp_df.replace('1;2', '1') # for exp/followup, 1 (past week) is more specific than 2 (since last visit)
        temp_df = temp_df.replace(['2', '1'], ['4', '2']) # need to remap past week to '2' and create code '4' for since last visit

    temp_df = temp_df.replace(['1;3', '2;3'], ['1', '2']) # if never and some other option are checked, assume the other option

    temp_df['version_form'] = 'expert' if 'expert' in form else 'parent'
    return temp_df

def update_ygtss(temp_df, form):
    temp_df['version_form'] = 'past week' if 'past_week' in form else 'post-drz'
    return temp_df


def split_multiform_row(form_dd_df, nih_form, form_df, update_func):
    nih_dd_df = pd.read_csv(join('nih_dd', nih_form + '_definitions.csv'), usecols=['ElementName', 'Aliases'])
    result = None
    redcap_forms = form_dd_df['form'].unique()
    for form in redcap_forms:
        form_cols = [ col for col in form_df if col in form_dd_df[form_dd_df['form'] == form].index ]
        print('\t', form, len(form_cols))
        temp_df = form_df[form_cols + [ col for col in form_df if col not in form_dd_df.index.values ]]
        temp_df = temp_df.dropna(how='all', subset=form_cols) # remove rows that are all null for current pattern (helps with screen/12mo form split)
        # temp_df = temp_df.rename(columns = {
        #     col: nih_dd_df[nih_dd_df['Aliases'].str.contains(col, na=False)].iloc[0,0] for col in form_cols if col not in nih_dd_df['ElementName'].values
        # })
        for col in form_cols:
            if col  in nih_dd_df['ElementName'].values:
                continue
            try:
                temp_df = temp_df.rename(columns={col: nih_dd_df[nih_dd_df['Aliases'].str.contains(col, na=False)].iloc[0,0]})
            except Exception as e:
                print(col)
                raise e

        if update_func:
            temp_df = update_func(temp_df, form)
        temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()] # remove duplicate column names (see first comment: https://stackoverflow.com/a/27412913)
        result = pd.concat([result, temp_df], sort=False) if result is not None else temp_df
    return result


def fill_known_missing(form_df, form):
    if form == 'cbcl01':
        form_df[(form_df['demo_study_id'] == 'NT808') & (form_df['visit_date'] == '11/20/2018')][['cbcl_25', 'cbcl_90']] = 999
    elif form == 'cbcl1_501':
        form_df[(form_df['demo_study_id'] == 'NT825') & (form_df['visit_date'] == '01/29/2018')][['ycbcl_q74', 'ycbcl_q92']] = 999
        form_df[(form_df['demo_study_id'] == 'NT828') & (form_df['visit_date'] == '02/20/2018')][' ycbcl_q86'] = 999
    elif form == 'kbit_ii01':
        form_df[form_df['demo_study_id'] == 'NT818'][['kbit_iq', 'kbit_riddles_raw', 'kbit_verbal_plus_nonverbal', 'kbit_verbal_raw', 'kbit_verbal_standard']] = 999

    return form_df


def replace_staff_names(all_data_df):
    # replace study staff/doctor names with generic references
    subset = [ col for col in all_data_df.columns if col != 'subjectkey']
    all_data_df[subset] = all_data_df[subset].replace({
        '(Emily|Emily Bihun|ECB|Vicki|Vicki Martin|VM|Soyoung|Soyoung Kim|SK|Kevin|Kevin Black|KJB|Samantha|Samantha Ranck|SR|Mary|Jackie|Jackie Hampton|JH)': 'rater',
        '(Dr. ([A-Z][a-z]+){1,2})': 'doctor'
    }, regex=True)
    return all_data_df


def get_redcap_df(guid_df, fields=None):
    api_db_password = getpass('API db password:')
    r01_project = get_redcap_project('r01', api_db_password)
    nt_project = get_redcap_project('nt', api_db_password)
    r01_df = r01_project.export_records(format='df', fields=fields)
    nt_df = nt_project.export_records(fields=['visit_date', 'demo_sex', 'demo_dob'], format='df')
    all_data_df = merge_projects(nt_df, r01_df)
    all_data_df = all_data_df.dropna(how='all')
    all_data_df = all_data_df.join(guid_df, how='inner')
    all_data_df = all_data_df[all_data_df['incl_excl_eligible'] != 0] # remove rows for excluded participants

    # remove rows for subjects in nt 9-11.5mo group
    nt9_11_ids = [ x[0] for x in all_data_df[all_data_df['incl_excl_new_tics_grp'] == 2].index.tolist() ]
    all_data_df = all_data_df.drop(nt9_11_ids, level='demo_study_id')

    # convert gender codes & calculate interview age
    all_data_df[['demo_sex', 'demo_dob', 'incl_excl_grp']] = all_data_df.groupby('demo_study_id')[['demo_sex', 'demo_dob', 'incl_excl_grp']].apply(lambda x: x.ffill().bfill())
    all_data_df['demo_sex'] = all_data_df['demo_sex'].replace([0, 1], ['F', 'M'])
    all_data_df['visit_date'] = all_data_df['visit_date'].fillna(all_data_df['mo3fupc_date']) # 3 month visit doesn't have the usual visit_date col
    all_data_df['visit_date'] = pd.to_datetime(all_data_df['visit_date'])
    all_data_df['interview_age'] = (all_data_df['visit_date'] - pd.to_datetime(all_data_df['demo_dob'])).apply(lambda x: round(.0328767*x.days) if pd.notnull(x) else np.nan)
    all_data_df['visit_date'] = all_data_df['visit_date'].map(lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else x)

    return all_data_df

def convert_redcap_to_nih(form_mapping_key, data_dict=None, convert_forms=None, to_date=None):
    # guid_df = get_guid_df()
    guid_df = pd.read_csv('guids.csv', index_col=1).rename(columns={'GUID': 'subjectkey'}).drop(columns='Date')
    guid_df.index.names = ['demo_study_id']
    if not data_dict:
        data_dict = join(BASE_PATH, r'REDCap\NewTicsR01_DataDictionary_2018-11-09.csv')
    data_dict_df = pd.read_csv(data_dict, index_col=0)
    data_dict_df = data_dict_df.rename(columns={'Form Name': 'form', 'Field Type': 'type', 'Choices, Calculations, OR Slider Labels': 'choices'})
    data_dict_df = data_dict_df[['form', 'type', 'choices']]

    form_map_df = pd.read_excel(form_mapping_key, skiprows=[1,2], index_col=0, usecols=[1,2])
    nih_forms = np.unique(form_map_df.index.values)
    if convert_forms:
        nih_forms = [ form for form in convert_forms if form in nih_forms ]
    all_data_df = get_redcap_df(guid_df)

    drop_cols = [ col for pattern in WITHHOLD for col in all_data_df.columns if re.match(pattern, col) ]

    # convert all checkboxes to labels
    checkbox_fields = data_dict_df[data_dict_df['type'] == 'checkbox'].index
    for field in checkbox_fields:
        use_label = (field in CHECKBOX_TO_LABEL)
        if not field + '___1' in all_data_df.columns:
            continue
        all_data_df =  replace_checkbox_with_label(all_data_df, data_dict_df, field, use_label)

    # replace study staff/doctor names with generic references
    all_data_df = replace_staff_names(all_data_df)

    form_field_map = {
        'ndar_subject01': ['incl_excl_grp', 'demo_race'],
        'srs02': ['incl_excl_grp', 'demo_completed_by'],
        'cbcl1_501': ['incl_excl_grp'],
        'socdemo01': ['cbcl_grade_in_school']
    }

    for form in nih_forms:
        required_fields = [ 'subjectkey', 'visit_date', 'interview_age', 'demo_sex']

        upload_file = join('import_forms', form + '.csv')
        if exists(upload_file):
            continue

        with open(upload_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(re.match('(\w+)(\d{2}$)', form).groups()))

        redcap_forms = list(form_map_df.xs(form).values.flatten())
        print(form, redcap_forms)

        form_dd_df = data_dict_df[data_dict_df['form'].isin(redcap_forms)]
        form_cols = list(form_dd_df[form_dd_df['type'] != 'descriptive'].index.values)
        keep_cols = form_cols + required_fields
        if form in form_field_map:
            keep_cols += form_field_map[form]
        form_df = all_data_df[np.unique(keep_cols)].reset_index()
        #subset = [ col for col in form_cols if col not in required_fields+WITHHOLD ] # TODO: figure out why I pulled out required_fields -- because it will allow pre-R01 data in
        subset = [ col for col in form_cols if col not in WITHHOLD ]
        if form != 'tsp01': # since we're joining to a different csv for tsp + most subjects do not have new form data, we cannot drop here
            form_df = form_df.dropna(how='all', subset=subset)
        form_df = form_df.rename(columns={'demo_sex': 'gender'})

        # convert numbers to label where needed
        convert_cols = [ col for pattern in OTHER_LABELS_NEEDED for col in form_df.columns if re.match(pattern, col) ]
        for col in convert_cols:
            form_df =  replace_num_with_label(form_df, data_dict_df, col)

        event_name_renames = ['Screening', '3 Month Follow-up', '12 Month Follow-up'] + [ 'Clinical Follow-up ' + str(n) for n in range(1,5) ]
        keep_event_col =  ['endvisit01', 'ticscreener01', 'mvhsp01', 'ygtss01', 'tsp01']
        if form not in keep_event_col:
            form_df.drop(columns='redcap_event_name', inplace=True) # all forms other than endvisit do not need the 'redcap_event_name' index col
        else:
            form_df['redcap_event_name'] = form_df['redcap_event_name'].replace(
                ['screening_visit_arm_1', '3_month_follow_up_arm_1', '12_month_follow_up_arm_1'] + [ 'clinical_follow_up_arm_1' + l for l in ['', 'b', 'c', 'd'] ],
                event_name_renames
            )
            rename = 'version_form' if form in keep_event_col[-2:] else 'visit'
            form_df = form_df.rename(columns={'redcap_event_name': rename})

        print(form_df.columns)
        if to_date:
            form_df = form_df[pd.to_datetime(form_df['visit_date']) < to_date]

        # endvisit01
        #   choose parent that signed consent for visit_parents_present when both parents were present
        if form == 'endvisit01':
            mom_consent = ['NT818', 'NT833', 'NT835', 'NT839', 'NT841', 'NT843', 'NT844', 'NT881']
            dad_consent = ['NT822']
            form_df.loc[form_df['demo_study_id'].isin(mom_consent), 'visit_parents_present'] = 1 # both parents present; mom signed consent
            form_df.loc[form_df['demo_study_id'].isin(dad_consent), 'visit_parents_present'] = 2 # both parents present; dad signed consent

        # ndar_subject01
        #   set our dna sample type to saliva, change value of sample usability to string, set required variables about type of study,
        #   specify required phenotype description (taken from iec01 group requirement), set race to 'More than one race' if they checked
        #   multiple
        if form == 'ndar_subject01':
            form_df['sample_description'] = np.where(form_df['dna_sample_collected'] == 1, 'saliva', np.nan)
            form_df['dna_sample_usable'] = form_df['dna_sample_usable'].replace([1,2], ['Sample usable', 'Sample not usable'])
            form_df = form_df.assign(twins_study = 'No', sibling_study = 'No', family_study = 'No')
            conditions = [(form_df['incl_excl_grp'] == 1), (form_df['incl_excl_grp'] == 2), (form_df['incl_excl_grp'] == 3)]
            choices = ['Tics now, but developed them only in the past 6 months', 'Meets DSM-5 criteria for Tourettes', 'No history of tics']
            form_df['phenotype_description'] = np.select(conditions, choices, default='999')
            form_df['demo_race'] = form_df['demo_race'].where(~form_df['demo_race'].str.contains(';'), 'More than one race')

        # phenotype required
        #   some forms (i.e. srs02, ndar_subject01) require phenotype - for all such forms (except iec01 that's already mapped as numbers)
        #   we want to change the field name and use the concatenated labels
        if form != 'iec01' and 'incl_excl_grp' in keep_cols:
            form_df =  replace_num_with_label(form_df, data_dict_df, 'incl_excl_grp').rename(columns={'incl_excl_grp': 'phenotype'})

        # socdemo01
            # only report one race (store others in separate var), add 'years' to education string, swap hispanic codes, make sure education
            #   is within string limits
        if form == 'socdem01':
            form_df = form_df.apply(separate_multi_response, args=('demo_race', 'fsprg'), axis=1)
            form_df['demo_ethnicity'] = form_df['demo_ethnicity'].replace(['1', '2'], ['2', '1'])
            form_df['demo_mat_edu'] = form_df['demo_mat_edu'].apply(add_years_str).str.slice(0,20)
            form_df['demo_pat_edu'] = form_df['demo_pat_edu'].apply(add_years_str).str.slice(0,20)

        # bsmss01 (ses)
        #   only report partner info for 'primary residence' parent's partner, remove occ details because of PHI issues
        if form == 'bsmss01':
            form_df = form_df.assign(version_form = np.nan, bsmss03_spouse = np.nan, bsmss07_spouse = np.nan)
            form_df = form_df.apply(ses_primary_partner, axis=1)
            drop_cols += ['primary_residence', 'ses_detail_occ_mother', 'ses_detail_occ_father'] + [ col for col in form_df.columns if col.endswith('partner') ]

        # mab01 (maternal/birth history)
        #   remove 'unknown' code for apgar responses, convert the task ages to months, make gest_wks is an integer
        if form == 'mab01':
            form_df['matern_no_preg'] = form_df['matern_no_preg'].replace('unknown', '')
            form_df[['matern_gest_wks']] = form_df[['matern_gest_wks']].apply(age_to_units, args=(AgeUnits.WEEKS,), axis=1)
            form_df[['matern_nicu_days']] = form_df[['matern_nicu_days']].apply(age_to_units, args=(AgeUnits.DAYS,), axis=1)
            apgar_cols = [ col for col in form_df.columns if col.startswith('matern_apgar') ]
            form_df[apgar_cols] = form_df[apgar_cols].replace(11, np.nan)
            matern_age_tsks = [ 'matern_' + col for col in ['hld_head', 'rll_ovr', 'toy_reach', 'sat_up', 'fing_fed', 'crawl', 'pull_to_stand', 'walk', 'slf_fed', 'tlk_wrd_combo'] ]
            form_df[matern_age_tsks] = form_df[matern_age_tsks].apply(age_to_units, args=(AgeUnits.MONTHS,), axis=1)

        # tichist01 (family history)
        #   remove relative_ from fh columns to fit within character limit
        if form == 'tichist01':
            relative_renames = { col: col.replace('relative_', '') for col in form_df.columns if re.match('fh_\w+_other_relative_relationship_800', col) }
            form_df =  form_df.rename(relative_renames, axis=1)

        # srs02
        #   map question responses to actual values (not scores), create separate gender norm columns for scores and rename, add columns
        #   for who completed form (taken from demo_completed_by field)
        if form == 'srs02':
            srsq_cols = get_matching_cols(form_df, 'srs_800_q\d+')
            form_df[srsq_cols] = form_df[srsq_cols].apply(lambda x: x.str[0], axis=1)
            raw_score_cols = [ 'srs_awareness', 'srs_cognition', 'srs_communication', 'srs_motivation', 'srs_mannerisms', 'srs_total' ]
            tscore_cols = [ col + '_t' for col in raw_score_cols ]
            form_df =  gender_norm_srs(form_df, raw_score_cols, 'raw')
            form_df =  gender_norm_srs(form_df, tscore_cols, 'tscore')
            form_df['srs_score'] = np.nan
            form_df['respond_detail'] = form_df['demo_completed_by'].apply(mom_or_dad)
            form_df['respond'] = np.where(form_df['respond_detail'].isin([1,2]), 1, np.nan)
            form_df = form_df.drop(columns='demo_completed_by')

        # ticscreener01
        #   only report most specific value of symptom checklist timeframe (i.e. 'past week' if both 'past week' and 'lifetime' selected),
        #   ensure consistent values between different visit forms (i.e. make 'past week' always be 2, introduce new code for 'since last visit')
        if form == 'ticscreener01':
            form_df = form_df.drop(columns=['exp_specify_tics_2_pw'])
            form_df = split_multiform_row(form_dd_df, form, form_df, update_ticscreener)

        # pedsql01
        #   replace version numeric code with string representing version, decrement question scale to 0-4
        if form == 'pedsql01':
            q_cols = get_matching_cols(form_df, 'pedsql_\w*_\d')
            form_df[q_cols] = form_df[q_cols] - 1
            form_df['pedsql_version'] = form_df['pedsql_version'].apply(lambda x: 'Version for Ages ' + x if pd.notnull(x) else np.nan)

        # adhdrs01
        #   combine all ADHD rating scale forms into one and use column version form to differentiate lifetime/expert/parent
        if form == 'adhdrs01':
            form_df = split_multiform_row(form_dd_df, form, form_df, update_adhdrs)

        # mvhsp01
        #   recode completed_by, and record visit type under version form
        if form == 'mvhsp01':
            replace_cols = ['med_completed_by', 'med_completed_by_12mo']
            form_df[replace_cols] = form_df[replace_cols].replace(range(1,5), 8)
            form_df[replace_cols] = form_df[replace_cols].replace([5, 6], [17, 98])
            form_df = split_multiform_row(form_dd_df, form, form_df, None)
            print(form_df.columns)
            form_df['chf_05'] = form_df['chf_05'].str.slice(0,250)

        # puts01
        #   specify that we used 9-item PUTS
        if form == 'puts01':
            form_df['version_form'] = '9-item PUTS'

        # ygtss01
        #   remove '_past_week' to meet column length limits
        if form == 'ygtss01':
            long_cols = ['ygtss_past_week_unavailable_list', 'ygtss_past_week_expert_total_tic_score']
            form_df =  form_df.rename({ col: col.replace('past_week_', '') for col in form_df.columns if col in long_cols }, axis=1)
            form_df = split_multiform_row(form_dd_df, form, form_df, update_ygtss)


        # erd_tics01
        #   rename impairment col to meet char limit, remove code for blank, reverse-code ocd response
        if form == 'erd_tics01':
            form_df = form_df.rename({ 'expert_diagnosis_impairment_distress': 'expert_impairment_distress'}, axis=1)
            form_df.replace(4, np.nan, inplace=True)
            form_df['expert_diagnosis_ocd'].replace([1,2,3], [3,2,1], inplace=True)
            form_df['expert_diagnosis_onset_notes'] = form_df['expert_diagnosis_onset_notes'].str.slice(0,20)

        # cybocs01
        #   label visit/form type (i.e. worst ever, past week), recode 'past week' to always be 3, add unique code for 'since last visit'
        if form == 'cybocs01':
            form_df = form_df.drop(columns='comp_counting_12mo') # FIXME: column is meant to be descriptive; remove once we have updated data dict
            form_df = split_multiform_row(form_dd_df, form, form_df, update_cybocs)
            for col in [ col for col in form_df if 'spec' in col ]:
                form_df[col] = form_df[col].apply(lambda x: x[:100] if pd.notnull(x) else x)

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
            # get cbcl columns in order they appear in form, so we can use column ranges
            ordered_cbcl_columns = [ x for x in data_dict_df[data_dict_df['form'] == 'child_behavior_checklist_cbcl_ages_6_18'].index.tolist() if x in form_df.columns ]
            ordered_cbcl_columns = [ col for col in form_df.columns if col not in ordered_cbcl_columns ] + ordered_cbcl_columns
            form_df = form_df[ordered_cbcl_columns]
            form_df.loc[:, 'cbcl_1':'cbcl_113c']= form_df.loc[:, 'cbcl_1':'cbcl_113c'].replace([1,2,3], [0,1,2])
            form_df.loc[:, 'cbcl_number_of_sports':'cbcl_chores_well3'] = form_df.loc[:, 'cbcl_number_of_sports':'cbcl_chores_well3'].replace([1,2,3,4,5], [0,1,2,3,999])
            form_df.loc[:, 'cbcl_hobbies1_a':'cbcl_hobbies3_b'] = form_df.loc[:, 'cbcl_hobbies1_a':'cbcl_hobbies3_b'].replace(999, 9) # hobbies/activities has different NA code
            form_df.loc[:, 'cbcl_close_friends':'cbcl_disability'] =  form_df.loc[:, 'cbcl_close_friends':'cbcl_disability'].replace([1,2,3,4,5], [0,1,2,3,9])
            form_df['cbcl_times_a_week_friends'] = form_df['cbcl_times_a_week_friends'].replace([1,2,3], [0,1,2]) # after decrementing range, go back and adjust this specific col
            fill_cols = ['cbcl_56h'] + [ col for col in form_df.columns if re.match('cbcl_(raw|t|per)_(activities|social|school|totalcomp)', col) ]
            form_df[fill_cols] = form_df[fill_cols].fillna(999)
            form_df.loc[:, 'cbcl_sports1_a':'cbcl_sports3_b'] = form_df.loc[:, 'cbcl_sports1_a':'cbcl_sports3_b'].fillna(999) # sports only is marked as required...

            missing_cols = [ 'cbcl_' + col for col in ['emotional', 'sleep', 'pervasive', 'sct', 'ocd', 'ptsd', 'depresspr'] ]
            missing_cols = list(chain(*[ [col, col + '_raw'] for col in missing_cols ])) # need cols with and without '_raw' suffix
            for col in missing_cols:
                form_df[col] = 999

            drop_cols += [ 'cbcl_' + col for col in ['age', 'date', 'gender', 'race', 'ethnicity', 'grade_in_school', 'attending_school',
                'informant_gender', 'informant_relation', 'notes']] # dropping notes as cbcl_comments maps to same field and cbcl_notes is unused

        # cbcl1_501
        #   Recode 3 as 4 (NA), set unrecorded newer columns to unknown (999), cap concerns response
        if form == 'cbcl1_501':
            form_df['ycbcl_age_5_yn'] = form_df['ycbcl_age_5_yn'].replace(3, 4)
            for col in ['cbcl_depresspr', 'cbcl_depresspr_raw']:
                form_df[col] = 999

        # cptc01
        #   Rename cols to fit within col name limits
        if form == 'cptc01':
            rename_cols = [ col for col in form_df.columns if re.match('cpt_hit\w*_change_guideline$', col) ]
            form_df = form_df.rename(columns={ col: col.replace('cpt_', '') for col in rename_cols })

            form_df['cpt_type'] = np.where(form_df['interview_age'] >= 72, 1, 2) # set assessment to KCPT if age < 6

            guideline_cols = [ col for col in form_df.columns if col.endswith('guideline') ]
            form_df[guideline_cols] = form_df[guideline_cols].apply(lambda x: x.str.lower())
            guideline_cols.remove('cpt_hit_rt_guideline')
            form_df[guideline_cols] = form_df[guideline_cols].replace('within average range', 'within the average range')

        # sldc01
        #   Split adhd comb/inatt/hyper fields into separate upload files
        if form == 'sldc01':
            for idx, type in enumerate(['comb', 'inatt', 'hyper']):
                adhd_type_cols = [ col for col in form_df.columns if col.startswith('ksads5_adhd_' + type) ]
                adhd_type_df = form_df[adhd_type_cols + [ col for col in form_df.columns if not col.startswith('ksads5') ]]
                adhd_type_df.assign(sldc153=idx, sldc153c=idx)
                adhd_type_file = 'sldc01_adhd_{}.csv'.format(type)
                with open(adhd_type_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(list(re.match('(\w+)(\d{2}$)', form).groups()))
                adhd_type_df.to_csv(adhd_type_file, mode='a', index=False, float_format='%g')
                form_df = form_df.drop(columns=adhd_type_cols)

        # tsp01
        #   Combine with drz_output for summary metrics, remap drz_tics to comments for extra length (it's more often used as a general
        #   comment field anyways), drop completely empty rows after merge
        if form == 'tsp01':
            drz_score_cols = ['demo_study_id', 'version_form', 'session', 'condition', 'tic_freq', 'tsp_tfi', 'duration', 'data_file1', 'notes']
            drz_score_df = pd.read_csv(join(BASE_PATH, 'TSP', 'drz_output.csv'), skiprows=0, names=drz_score_cols)
            print(drz_score_df)
            drz_score_df['version_form'] = drz_score_df['version_form'].replace(['screen'] + [ str(d) + 'mo' for d in [3,12,24,36,48,60] ], event_name_renames)

            form_df = form_df.merge(drz_score_df, on=['demo_study_id', 'version_form'], how='left')
            form_df['int_dur'] = 10 # duration of tic-free intervals
            form_df['version_form'] = form_df[['version_form', 'condition', 'session']].apply(lambda x: '; '.join(x.astype(str)), axis=1)
            form_df = form_df.drop(columns=['condition', 'session', 'duration', 'notes'])
            form_df = form_df.rename(columns={'drz_tics': 'comments'})

            form_df = form_df.dropna(how='all', subset=[col for col in subset + ['tic_freq', 'tsp_tfi', 'data_file1'] if col in form_df])

        # convert all text-field ages to months
        age_to_months_cols = [ col for col in form_df.columns if re.match('(matern|ksads|fh)(_\w*)*_age', col) and \
            col not in list(form_dd_df[form_dd_df['type'] == 'radio'].index.values) ]
        if age_to_months_cols:
            # assume ages (other than ones in mab01 which ask for intended unit) will be reported in years
            form_df[age_to_months_cols] = form_df[age_to_months_cols].apply(age_to_units, args=(AgeUnits.MONTHS, AgeUnits.YEARS), axis=1)

        # remove pre-R01 rows
        print(form_df['visit_date'])
        form_df = form_df[pd.to_datetime(form_df['visit_date']) > datetime(2017, 8, 1)]

        form_df = fill_known_missing(form_df, form)

        # write out data to separate NIH form files
        form_df = form_df.drop(columns=drop_cols + [col for col in form_df.columns if col.endswith('_complete')], errors='ignore')
        form_df.to_csv(upload_file, mode='a', index=False, float_format='%g')

    return


@Gooey()
def parse_args():
    parser = GooeyParser()
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--form_mapping_key', required=True, widget='FileChooser', help='NIH form mapping key excel file')

    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--data_dictionary', widget='FileChooser', help='most recent REDCap data dictionary')
    optional.add_argument('-f', '--form', nargs='+', help='NIH form(s) to convert (default is all)')
    optional.add_argument('--to_date', widget='DateChooser', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='only process subjects up until date')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_redcap_to_nih(args.form_mapping_key, args.data_dictionary, args.form, args.to_date)
