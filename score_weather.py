# modified version of script in jupyter notebook (KJB\Weather_Prediction_scoring.ipynb)

from chardet.universaldetector import UniversalDetector
from gooey import Gooey, GooeyParser
from os import listdir, stat
from os.path import join
import pandas as pd
import re

WEATHER_DIRECTORY = '../../../Raw_Data/weather'
OUTPUT_FILE = 'weather_result.csv'
TOTAL_TRIALS = 90
INDEX_COLS = ['demo_study_id']


def theor_correct(n): # n is the CueId
	"""Return character code for Theoretically Correct response for the CueId supplied as a string"""
	# alternatively, supply as binary number, e.g. if n==0b0110:
	if (n=='0001' or n=='0010' or n=='0011' or n=='0101' or n=='0111' or n=='1011'):
		return 'g'
	if (n=='0100' or n=='1000' or n=='1001' or n=='1010' or n=='1100' or n=='1101' or n=='1110'):
		return 'h'
	if n=='0110':
		return 'x'
	assert True, "unexpected CueId: {0}".format(n)

def score_weather(dir=WEATHER_DIRECTORY, use_existing=False):
	summary_cols = ['weather_all']
	if use_existing:
		return pd.read_csv(join(dir, OUTPUT_FILE)).set_index(INDEX_COLS)[summary_cols]

	filename_format = 'weather-\d{2,3}-1.txt'
	files = [ f for f in listdir(dir) if re.match(filename_format, f, flags=re.IGNORECASE) ]
	results = [] # array of arrays of block percentages for each subject
	detector = UniversalDetector() # files are a mix of utf-8 and utf-16-le
	for filename in files:
		past_header = False
		cueid = ''
		trial = 1     # numbered from 1 in the input data
		blocknum = 0  # first block, number 0
		scores = []
		subject = 'Unknown Subject'
		correctnums   = [0]
		correctdenoms = [0]
		correctpcts   = [0]
		print("Reading filename {0}".format(filename))

		detector.reset()
		with open(join(dir, filename), 'rb') as f:
			if stat(f.name).st_size == 0:
				continue # skip if file is empty (do not try to calculate score, do not create empty result row)

			for line in f:
				detector.feed(line)
				if detector.done:
					break
			detector.close()

		with open(join(dir, filename), encoding=detector.result['encoding']) as f:
			for line in f:
				linewords = line.split()
				if not past_header:         # deal with header
					if "Subject:" in line:
						# some of the 7## participants have subject numbers appearing without the 7-prefix
						# 	add it here if not already 3 digits
						subject = 'NT' + (linewords[-1] if len(linewords[-1]) == 3 else '7' + linewords[-1])
						past_header = True
						print(subject)
					else:
						continue  # (the loop 'for line in f')
				# now we're past the header
				assert trial < 2+TOTAL_TRIALS, "trial {0} > {1}".format(trial,1+TOTAL_TRIALS)
				if trial > 50 :
					procedure = 2
					cue_number = trial-50
				else:
					procedure = 1
					cue_number = trial
				# Process one frame (trial)
				if 'LogFrame End' in line:
					if trial % 10 == 0:  # i.e. number of the trial just processed ends in 0
						# don't divide by zero
						assert correctdenoms[blocknum] > 0, "No usable trials in block {0}".format(blocknum)
						correctpcts[blocknum] = correctnums[blocknum] / correctdenoms[blocknum]
						if trial < TOTAL_TRIALS:  # we need another list entry to add data to
							correctnums.append(0)
							correctdenoms.append(0)
							correctpcts.append(-1)  # so we can tell if something went wrong
						blocknum = trial // 10
					trial += 1
					cueid = ''  # see assertion below with "got to response in trial {0} without reading CueId"
				elif 'CueList'+str(procedure)+':' in line:
					frame = linewords[-1]
					assert frame == str(cue_number), \
						"frame {0} doesn't equal reported frame {1} in {2}".format(frame, cue_number, line)
				elif 'CueId:' in line:
					cueid = linewords[-1]
				elif 'CueImage1.RESP:' in line:
					assert cueid != '', "got to response in trial {0} without reading CueId".format(trial)
					response = linewords[-1]
					if theor_correct(cueid) == 'x':
						scores.append('X')
					elif response not in 'gh':  # no answer that I can read
						scores.append('N')
					elif response == theor_correct(cueid):  # response is "theoretically correct" (see Word document)
						scores.append('1')
						correctnums[blocknum] += 1
						correctdenoms[blocknum] += 1
					else: # incorrect response
						scores.append('0')
						correctdenoms[blocknum] += 1
		print(correctnums, correctdenoms)
		overall_frac_theor_correct = sum(correctnums)/sum(correctdenoms)
		correctpcts = [ 100 * pct for pct in correctpcts ]
		results.append([subject, ''.join(scores), 100*overall_frac_theor_correct] + correctpcts)

	columns = ['demo_study_id', 'weather_scores', 'weather_all'] + [ 'weather_block' + str(i) for i in range(1,10) ]
	df = pd.DataFrame(data = results, columns = columns).set_index(INDEX_COLS)
	df.to_csv(join(dir, OUTPUT_FILE))
	return df[summary_cols]


if __name__ == '__main__':
	@Gooey
	def parse_args():
		parser = GooeyParser()
		parser.add_argument('indir', widget='DirChooser', help='directory containing weather txt files')
		return parser.parse_args()


	args = parse_args()
	score_weather(args.indir)
