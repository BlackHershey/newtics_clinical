import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # must be set prior to pyplot import
from matplotlib import pyplot

df = pd.read_csv('all_data_result.csv')
#tsp_deltas = (df['avg_tics_per_minute_verbal_screen'] - df['avg_tics_per_minute_baseline_screen']).values
#tsp_deltas = (df['avg_tics_per_minute_DRZ_screen'] - df['avg_tics_per_minute_baseline_screen']).values
#tsp_deltas = (df['avg_tic_free_10s_per_minute_verbal_screen'] - df['avg_tic_free_10s_per_minute_baseline_screen']).values
tsp_deltas = (df['avg_tic_free_10s_per_minute_DRZ_screen'] - df['avg_tic_free_10s_per_minute_baseline_screen']).values

ygtss_deltas = (df['ygtss_past_week_expert_total_tic_12mo'] - df['ygtss_past_week_expert_total_tic_screen']).values

nans = np.unique(np.concatenate((np.argwhere(np.isnan(tsp_deltas)), np.argwhere(np.isnan(ygtss_deltas)))))
tsp_deltas = np.delete(tsp_deltas, nans)
ygtss_deltas = np.delete(ygtss_deltas, nans)

pyplot.scatter(tsp_deltas, ygtss_deltas)
pyplot.plot(np.unique(tsp_deltas), np.poly1d(np.polyfit(tsp_deltas, ygtss_deltas, 1))(np.unique(tsp_deltas)))
pyplot.xlabel('TSP Screen DRZ-Baseline delta')
pyplot.ylabel('YGTSS Total Tic Score 12mo-Screen delta')
print('r = {}'.format(np.corrcoef(tsp_deltas, ygtss_deltas)[1,0]))
print("# of subjects", len(tsp_deltas))
pyplot.show()
