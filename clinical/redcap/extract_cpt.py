import csv
import itertools
import re
import pandas as pd
from gooey import Gooey, GooeyParser
from os import listdir
from os.path import isfile, join, splitext

CPT_DIRECTORY = '../../../Raw_Data/CPT'
OUTPUT_FILE = 'cpt_result.csv'
INDEX_COLS = ['demo_study_id']


measures_with_pct = ['Omissions', 'Commissions', 'Perseverations']
measures_without_pct = ['Hit RT', 'Hit RT Std. Error', 'Variability', 'Detectability', 'Response Style', 'Hit RT Block Change',
	'Hit SE Block Change', 'Hit RT ISI Change', 'Hit SE ISI Change']

def extract_cpt(indir=CPT_DIRECTORY, use_existing=False):
	# generate column names by combining each measure with each attribute (excluding hit_rt x pct)
	all_measures = measures_with_pct + measures_without_pct
	all_measures.remove('Perseverations')
	all_measures.insert(7, 'Perseverations') # put column list in order they appear in CPT file
	all_measures = [ s.lower().replace('.', '').replace(' ', '_') for s in all_measures ]
	attributes = [ 'n', 't', 'pctile', 'guideline', 'pct' ]
	#columns = ['demo_study_id' , 'cpt_type'] + [ 'cpt_' + '_'.join([a,b]) for a, b in list(itertools.product(all_measures, attributes)) ]
	columns = ['demo_study_id', 'cpt_type'] + [ 'cpt_' + '_'.join([a,b]) for a, b in list(itertools.product(all_measures, attributes)) ]

	attributes.remove('guideline')
	summary_columns = [ col for col in columns if re.match('cpt_(omissions|commissions|hit_rt)_(' + '|'.join(attributes) + ')', col) ]
	summary_columns.remove('cpt_hit_rt_pct')

	if use_existing:
		return pd.read_csv(join(indir, OUTPUT_FILE)).set_index(INDEX_COLS)[summary_columns]

	filename_format = '((NT\d{3})_(K?CPT)\w*.xlsx?)' # matches files beginning with NT<id>_KCPT and NT<id>_CPT
	search_res = [ re.match(filename_format, f, flags=re.IGNORECASE) for f in listdir(indir) ]
	excel_files = [ s.groups() for s in search_res if s ]

	results = [] # array of arrays of CPT omission/commission/hit rt data for each subject
	for (filename, subject, cpt_type) in excel_files:
		print('Extracting', filename)
		file_contents = pd.read_excel(join(indir, filename)).values.tolist()

		cpt_type = cpt_type.upper()
		if cpt_type == 'CPT':
			cpt_type += '-II'

		one_more = False
		result_row = [subject, cpt_type]
		for row in file_contents:
			row = [ str(x) for x in row ]
			if one_more: # then next line should be %
				one_more = False
				result_row.append(row[1])
				continue
			if row and any(row[0].startswith(measure) for measure in measures_with_pct):
				result_row += row[1:5]
				one_more = True
			elif row and any(row[0].startswith(measure) for measure in measures_without_pct):
				result_row += row[1:5]
				result_row.append(None)

			if 'Summary of Inattention Measures' in row:
				break
		results.append(result_row)

	df = pd.DataFrame(data = results, columns = columns).set_index(INDEX_COLS)
	df = df.dropna(axis=1, how='all') # drop columns that are all NaN measues that don't apply to all labels
	df = df.dropna(axis=0, how='all') # drop rows where assessment was not completed
	df = df.assign(cpt_scored_data_complete=2)
	df.to_csv(join(indir, OUTPUT_FILE))

	return df[summary_columns]


if __name__ == '__main__':
	@Gooey
	def parse_args():
		parser = GooeyParser()
		parser.add_argument('indir', widget='DirChooser', help='directory containing cpt excel files')
		return parser.parse_args()


	args = parse_args()
	extract_cpt(args.indir)
