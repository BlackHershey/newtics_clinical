import pandas as pd
import pyodbc

from getpass import getuser
from redcap import Project, RedcapError

DB_PATH_TEMPLATE = r'K:\Projects\TS\NewTics\api_tokens.accdb'
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
		exit('Error connecting to API token access database')

	cursor = conn.cursor()
	sql = 'SELECT api_token FROM {}_api_tokens WHERE userid = ?'.format(study_name)
	cursor.execute(sql, (user,))
	api_token = cursor.fetchone()[0]
	return Project(REDCAP_URL, api_token)

def get_project_df(project_name, datafile=None, api_db_password=None, fields=None):
	if datafile:
		print('GET_PROJECT_DF FOR {} USING CSV {}'.format(project_name,datafile))
		df = pd.read_csv(datafile, index_col=[0,1])
		if fields:
			df = df[fields]
	else:
		print('PULLING DATA FROM REDCAP FOR {}'.format(project_name))
		redcap_project = get_redcap_project(project_name, api_db_password)
		df = redcap_project.export_records(fields=fields, format='df')
	return df

