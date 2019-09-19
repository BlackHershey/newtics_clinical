import argparse
import numpy as np
import pandas as pd
import pyodbc
import re

from getpass import getpass

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
		sys.stderr.write('Error connecting to access database')
		sys.exit(1)

	sql = 'SELECT Contacts.FirstName, Contacts.LastName, Contacts.Guardian1name, Contacts.Guardian2name, Contacts.Birthdate' + \
			' FROM Contacts' + \
			' WHERE Contacts.NewTicsID IS NOT NULL';
	return pd.read_sql(sql, conn)


def search_data(row, search):
	for str in row:
		print(str)
		return bool(pd.notnull(str) and re.search(search, str))


def flag_redcap_phi(study_key):
	password = getpass('db password:')
	project = get_redcap_project(study_key, password)
	df = project.export_records(format='df').dropna(how='all', axis=0).drop(columns='demo_dob')

	lut = get_phi_lut(password)
	lut = lut.drop(columns='Birthdate').fillna('')
	lookup_string = '|'.join({ x for x in filter(str.isalpha, lut.to_string().split()) })

	print('beginning checks...')
	for col in df.columns:
		#df[col] = df[col].astype(str).str.contains(lookup_string, case=False, na=False, regex=True)
		df[col] = df[col].apply(lambda x: x if pd.notnull(x) and re.search(r'\b({})\b'.format(lookup_string), str(x), flags=re.IGNORECASE) else np.nan)
	df.dropna(how='all', axis=1).dropna(how='all', axis=0).to_csv('{}_phi_issues.csv'.format(study_key))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('study', choices=['nt','r01'])
	args = parser.parse_args()

	flag_redcap_phi(args.study)
