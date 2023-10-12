from modules.dataset import phosc_dataset

a = phosc_dataset('train.csv', 'dte2502_ga01_small')
#print column names
print(a.df_all.columns)
