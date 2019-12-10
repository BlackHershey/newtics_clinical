import pandas as pd
import pyodbc

from getpass import getuser
from redcap import Project, RedcapError

DB_PATH_TEMPLATE = r'\\neuroimage\nil\blackf\K\Projects\TS\NewTics\api_tokens.accdb'
REDCAP_URL = 'https://redcap.wustl.edu/redcap/srvrs/prod_v3_1_0_001/redcap/api/'

def get_redcap_project(study_name, password):
	user = getuser()
	try:
		conn_str = (
			r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=' + DB_PATH_TEMPLATE.format(user) + ';'
			r'PWD=' + password
		)
		conn = pyodbc.connect(conn_str)
	except pyodbc.Error:
		exit('Error connecting to access database')

	cursor = conn.cursor()
	sql = 'SELECT api_token FROM {}_api_tokens WHERE userid = ?'.format(study_name)
	cursor.execute(sql, (user,))
	api_token = cursor.fetchone()[0]
	return Project(REDCAP_URL, api_token)


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


def get_project_df(project_name, datafile=None, api_db_password=None, fields=None):
	if datafile:
		df = pd.read_csv(datafile, index_col=[0,1])
		if fields:
			df = df[fields]
	else:
		redcap_project = get_redcap_project(project_name, api_db_password)
		df = redcap_project.export_records(fields=fields, format='df')
	return df

