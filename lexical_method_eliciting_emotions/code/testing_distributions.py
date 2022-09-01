
import pandas as pd
pd.set_option("max_colwidth", 320)
import numpy as np

def load_data(file_path, indx = True, indx_col = 0):
  '''Parameters:
  file_path: path to your excel or csv file with data,
  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True
  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)

  return data



#testing distribution differences between affective databases and our data
#example here presented for Emean lexicon and Czas Decyzji Twitter dataset

data = load_data("/content/drive/MyDrive/Colab Notebooks/twitter/NAWL_EMEAN_2nd_round/tweets_Debata_CzasDecyzji_Emean_AVG_CAT.xlsx")
print(len(data))

cols = [
        'Happiness_individual_values', 'Anger_individual_values',
       'Sadness_individual_values', 'Fear_individual_values',
       'Disgust_individual_values', 'Valence_individual_values',
       'Arousal_individual_values', 'Surprise_individual_values',
       'Trust_individual_values', 'Anticipation_individual_values'
]

deb_to_analyze = data[cols]

from ast import literal_eval
deb_to_analyze = deb_to_analyze.applymap(literal_eval)

hap_full_arg_ind = np.concatenate(deb_to_analyze["Happiness_individual_values"].to_numpy()).ravel().tolist()
ang_full_arg_ind = np.concatenate(deb_to_analyze["Anger_individual_values"].to_numpy()).ravel().tolist()
sad_full_arg_ind = np.concatenate(deb_to_analyze["Sadness_individual_values"].to_numpy()).ravel().tolist()
fea_full_arg_ind = np.concatenate(deb_to_analyze["Fear_individual_values"].to_numpy()).ravel().tolist()
dis_full_arg_ind = np.concatenate(deb_to_analyze["Disgust_individual_values"].to_numpy()).ravel().tolist()
val_full_arg_ind = np.concatenate(deb_to_analyze["Valence_individual_values"].to_numpy()).ravel().tolist()
aro_full_arg_ind = np.concatenate(deb_to_analyze["Arousal_individual_values"].to_numpy()).ravel().tolist()
ant_full_arg_ind = np.concatenate(deb_to_analyze["Anticipation_individual_values"].to_numpy()).ravel().tolist()
tru_full_arg_ind = np.concatenate(deb_to_analyze["Trust_individual_values"].to_numpy()).ravel().tolist()
sur_full_arg_ind = np.concatenate(deb_to_analyze["Surprise_individual_values"].to_numpy()).ravel().tolist()
print(len(aro_full_arg_ind))


em_baza = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Emotional word lists/emean_emo_cat.xlsx", index_col=0)
em_baza = em_baza[['lemma', 'VAL M', 'ARO M', 'ANG M', 'DIS M', 'FEA M', 'SAD M', 'ANT M', 'HAP M', 'SUR M', 'TRU M']]
print(len(em_baza))



#######   desity distribution overlap function   ##########

def overlap(data_corpus, data_lexicon):
  '''
  x0: arrray of values from data (debate corpus),
  x1: baza["nazwa kolumny z emocją"]
  '''
  from matplotlib import pyplot as plt
  from scipy.stats import gaussian_kde

  x0 = pd.Series(data_corpus)
  x1 = pd.Series(list(data_lexicon.values))

  kde0 = gaussian_kde(x0, bw_method=0.3)
  kde1 = gaussian_kde(x1, bw_method=0.3)

  xmin = min(x0.min(), x1.min())
  xmax = max(x0.max(), x1.max())
  dx = 0.2 * (xmax - xmin) # add a 20% margin, as the kde is wider than the data
  xmin -= dx
  xmax += dx

  plt.figure(figsize=(10, 7))
  x = np.linspace(xmin, xmax, 500)
  kde0_x = kde0(x)
  kde1_x = kde1(x)
  inters_x = np.minimum(kde0_x, kde1_x)

  plt.plot(x, kde0_x, color='b', label='corpus')
  plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
  plt.plot(x, kde1_x, color='orange', label=data_lexicon.name)
  plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
  plt.plot(x, inters_x, color='r')
  plt.fill_between(x, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx', label='intersection')
  plt.legend()
  area_inters_x = np.trapz(inters_x, x)
  print(f"Overlap in {data_lexicon.name}: {round(area_inters_x, 5)}")



# emean
data_vars = [hap_full_arg_ind, ang_full_arg_ind, sad_full_arg_ind, fea_full_arg_ind, dis_full_arg_ind,
             val_full_arg_ind, aro_full_arg_ind, ant_full_arg_ind, tru_full_arg_ind, sur_full_arg_ind]
aff_vars = ['HAP M', 'ANG M', 'SAD M', 'FEA M', 'DIS M', 'VAL M', 'ARO M', 'ANT M', 'TRU M', 'SUR M']


for var in zip(aff_vars, data_vars):
  overlap(var[1], em_baza[var[0]])


import scipy
from scipy.stats import mannwhitneyu, normaltest, ttest_ind, ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import sqrt
plt.style.use("seaborn-talk")

def test_distributions(aff_db_value, data_indiv_values):

  print(f"Testing for variable --> {aff_db_value.name} <--")
  print()
  print("1) testing distributions' normality...")
  print(f"for affective database:    {normaltest(aff_db_value)} ")
  print(f"for values in your data:   {normaltest(data_indiv_values)} ")
  print()
  print("****     check p-values and then decide which statistic to take !!!     ****")
  print()

  print("2) testing statistics for t-test independent variables - take this if variables' distributions ARE normal")
  print(ttest_ind(aff_db_value, data_indiv_values, equal_var = False))
  print()

  print("3) testing statistics for U Manna-Whitney'a - take this if variables' distributions are NOT normal")
  print(mannwhitneyu(aff_db_value, data_indiv_values))
  print()


  print("4) two-sample Kolmogorov-Smirnov test for goodness of fit --> that could be even better statistic")
  print(ks_2samp(aff_db_value, data_indiv_values))
  print()

  cohens_d = (mean(aff_db_value) - mean(data_indiv_values)) / (sqrt((stdev(aff_db_value) ** 2 + stdev(data_indiv_values) ** 2) / 2))
  print("5) Cohen's d    -->  measure of effect size (small, medium, large):")
  print(cohens_d)
  print()


for var in zip(aff_vars, data_vars):
  test_distributions(em_baza[var[0]], var[1])
  print()
  print()
