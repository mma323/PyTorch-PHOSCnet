from modules.dataset import phosc_dataset

a = phosc_dataset('train.csv', 'dte2502_ga01_small')

print(a.df_all[1:5])