import numpy as np
import pandas as pd
import pyodbc
import re
import sys
sys.path.append(r'C:\Users\acevedoh\Box\Black_lab\projects\TS\NewTics\Data\analysis\clinical_data\scripts')

from getpass import getpass
from score_redcap_data import get_redcap_project

DB_PATH = r'K:\Projects\TS\NewTics\Recruitment\NewTics_recruit_forms.accdb'

def get_phi_lut(password):
	try:
		conn_str = (
			r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=' + DB_PATH + ';'
			r'PWD=' + password
		)
		conn = pyodbc.connect(conn_str)
	except pyodbc.Error:
		stderr.write('Error connecting to access database')
		exit(1)

	sql = 'SELECT Contacts.FirstName, Contacts.LastName, Contacts.Guardian1name, Contacts.Guardian2name, Contacts.Birthdate' + \
			' FROM Contacts' + \
			' WHERE Contacts.NewTicsID IS NOT NULL';
	return pd.read_sql(sql, conn)


def search_data(row, search):
	for str in row:
		print(str)
		return bool(pd.notnull(str) and re.search(search, str))

def flag_redcap_phi():
	password = getpass('db password:')
	r01_project = get_redcap_project('r01', password)
	df = r01_project.export_records(format='df').dropna(how='all', axis=0).drop(columns='demo_dob')

	lut = get_phi_lut(password)
	dob = lut['Birthdate']
	lut = lut.drop(columns='Birthdate').fillna('')
	lookup_string = '|'.join({ x for x in filter(str.isalpha, lut.to_string().split()) })
	#lookup_string += '|Dr\.|Vicki|Martin|Emily|Bihun|KJB|ECB|VM'

	print('beginning checks...')
	for col in df.columns:
		#df[col] = df[col].astype(str).str.contains(lookup_string, case=False, na=False, regex=True)
		df[col] = df[col].apply(lambda x: x if pd.notnull(x) and re.search(lookup_string, str(x)) else np.nan)
	df.dropna(how='all', axis=1).dropna(how='all', axis=0).to_csv('phi_issues.csv')


flag_redcap_phi()
