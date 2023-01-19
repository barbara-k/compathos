
# python -m streamlit run C:\Users\user1\Downloads\folder\app.py


# imports
import streamlit as st
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-talk")

import spacy
nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# functions

ethos_mapping = {0: 'neutral', 1: 'support', 2: 'attack'}
valence_mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}


def clean_text(df, text_column):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df["content"] = new_texts
  return df


#def wordcloud_lexeme(dataframe, lexeme_threshold = 90, count_threshold = 2, analysis_for = 'support', cmap_wordcloud = 'crest'):
  #dataframe['precision'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  #dfcloud = dataframe[(dataframe['precision'] >= int(lexeme_threshold)) & (dataframe['general #'] > int(count_threshold)) & (dataframe.word.map(len)>3)]
  #dfcloud = dfcloud.sort_values(by = 'precision', ascending = False)
  #words = dfcloud['word'].unique()
  #return words


import random
def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'Greens'):

  dataframe['precis'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp
  dfcloud = dataframe[(dataframe['precis'] >= int(lexeme_threshold)) & (dataframe['general #'] > 2) & (dataframe.word.map(len)>3)]
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    if not w.startswith('@'):
      w = str(w).strip()
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
      l = np.repeat(w, n)
      text.extend(l)
  random.shuffle(text)
  st.write(f"There are {n_words} words.")
  if n_words < 1:
      st.error('No words with a specified threshold. \n Choose lower value.')
      st.stop()
  figure_cloud, words_wcl = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud, words_wcl




def prepare_cloud_lexeme_data(data_neutral, data_support):
  # neutral df
  neu_text = " ".join(data_neutral['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  df2 = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2


def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                background_color ='black',
                min_font_size = 16, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig, wordcloud.words_.keys()



def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


@st.cache(allow_output_mutation=True)
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


def lemmatization(dataframe, text_column):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].apply(str)):
    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 2) ]))
  df[text_column +"_lemmatized"] = lemmas
  return df


def clean_text(df, text_column):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df["content"] = new_texts
  return df









import time

##################### page config  #####################
st.set_page_config(page_title="EmAn", layout="wide") # centered wide

#####################  page content  #####################
st.title("Annotation of Expressed Emotions Analysis")
add_spacelines(1)




import glob
dfs = glob.glob(r"C:\Users\user1\Downloads\folder\JEK\*.xlsx")

df_all = pd.DataFrame(columns = ['full_text_id', 'conversation_id', 'source', 'full_text', 'sentence',
       'joy', 'fear', 'anger', 'sadness', 'disgust', 'trust', 'suma', 'filled'])
for p in dfs:
  df0 = load_data(p)
  df0 = df0.iloc[:-1]

  cols_emo_renam = {
  'radość':'joy',
  'strach':'fear',
  'złość':'anger',
  'złosć':'anger',
  'smutek':'sadness',
  'wstręt':'disgust',
  'zaufanie':'trust'}
  df0 = df0.rename(columns = cols_emo_renam)
  try:
    p = int(str(p[18:-5]).replace('_', ''))
  except:
    p = str(p[18:-5]).replace('_', '')
  df0['ann'] = p
  df_all = pd.concat([df_all, df0], axis=0)

df_all = df_all.reset_index()
df_all = df_all.rename(columns={'index':'id_data'})
df_all.conversation_id = df_all.conversation_id.astype('str')

data1 = load_data(r"C:\Users\user1\Downloads\EthApp-main\tweets_CovidVaccine_sents2.xlsx")
split_dict = {}
low = 0
high = 910
for i in range(4):
  if i != 3:
    dd = data1.iloc[low:high]
    dd = dd.reset_index()
    dd = dd.rename(columns={'index':'id_data'})
    split_dict[i+1] = dd
    low += 910
    high += 910
  else:
    dd = data1.iloc[low:]
    dd = dd.reset_index()
    dd = dd.rename(columns={'index':'id_data'})
    split_dict[i+1] = dd


cols_emo = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust']
cols_emo1 = ['positive', 'negative_1', 'negative_2', 'neutral']
cols_emo2 = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust', 'positive', 'negative_1', 'negative_2', 'neutral', 'contains_emotion']
cols_emo_bin = ['neutral', 'contains_emotion']
df_all = df_all.fillna(0)

df_all['positive'] = np.where( (df_all[['joy', 'trust']].any(axis=1)), 1, 0)
df_all['negative_1'] = np.where( (df_all[['fear', 'sadness', 'anger', 'disgust']].any(axis=1)), 1, 0)
df_all['negative_2'] = np.where( (df_all[['anger', 'disgust']].any(axis=1)), 1, 0)
df_all['neutral'] = np.where( ~(df_all[cols_emo].any(axis=1)), 1, 0)
df_all['contains_emotion'] = np.where( df_all['neutral'] == 0, 1, 0)


# remove str chcracters
for c in cols_emo:
  ids = df_all[~(df_all[c].isin([0,1]))].index
  if len(ids) > 0:
    df_all.loc[ids, c] = 1


s1 = split_dict[1]['id_data'].unique()
s2 = split_dict[2]['id_data'].unique()
s3 = split_dict[3]['id_data'].unique()
s4 = split_dict[4]['id_data'].unique()

anns_size = df_all.groupby('ann', as_index=False)['filled'].mean().round(2)
anns_size = anns_size[ (anns_size.filled < 0.90) & (anns_size.filled > 0.15) ]['ann'].values




#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Parameters of analysis")
    add_spacelines(1)

    contents_radio_type = st.radio("Choose data", ('All annotations', 'Selected annotations'))
    add_spacelines(1)

    if contents_radio_type == 'Selected annotations':
        # selected
        df_all = df_all[ df_all.ann.isin(anns_size) ]

    page_content = st.radio("Analytics", ('IAA results', 'Distribution', 'Wordclouds', 'Explore cases'))
    add_spacelines(1)


#  *********************** sidebar  *********************



df_all[cols_emo2] = df_all[cols_emo2].astype('int').fillna(0)

df1 = df_all[df_all.id_data.isin(s1)]
df2 = df_all[df_all.id_data.isin(s2)]
df3 = df_all[df_all.id_data.isin(s3)]
df4 = df_all[df_all.id_data.isin(s4)]

cols_emo = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust']
cols_emo1 = ['positive', 'negative_1', 'negative_2', 'neutral']
cols_emo2 = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust', 'positive', 'negative_1', 'negative_2', 'neutral']
cols_emo_bin = ['neutral']

dfs2 = {1:df1, 2:df2, 3:df3, 4:df4}

if page_content == 'IAA results':
    sampl = []
    emo = []
    fle = []
    fle_adj = []
    accs_perf = []
    accs_maj = []

    for k in list(dfs2.keys()):
      dd = dfs2[k]

      for e in cols_emo2:
        ddict = {}
        if dd.ann.nunique() > 2:
            for a2 in dd.ann.unique():
              s2 = dd[dd.ann == a2].sort_values(by = 'id_data')[[e]].astype('int').reset_index(drop=True) #.astype('int').values
              ddict[a2] = s2

            dim = pd.concat(list(ddict.values()), axis=1, ignore_index=True)
            agg = aggregate_raters(dim.to_numpy())[0]

            num_ann = len(dd.ann.unique())
            dim['acc_iaa'] = dim.sum(axis=1)
            acc_agg_maj = round(dim[ (dim.acc_iaa > np.ceil(num_ann/2)) | (dim.acc_iaa < np.floor(num_ann/2)) ].shape[0] / dim.shape[0], 3)
            acc_agg_perf = round(dim[(dim.acc_iaa == 0) | (dim.acc_iaa == num_ann)].shape[0] / dim.shape[0], 3)
            accs_maj.append(acc_agg_maj)
            accs_perf.append(acc_agg_perf)

            #if (num_ann % 2) != 0:
              #dim['fl_adj_1'] = np.where(dim.acc_iaa >= np.ceil(num_ann/2), num_ann, 0)
              #dim['fl_adj_0'] = np.where(dim.acc_iaa <= np.floor(num_ann/2), num_ann, 0)
            #else:
                #dim['fl_adj_1'] = np.where(dim.acc_iaa > np.ceil(num_ann/2), num_ann, 0)
                #dim['fl_adj_0'] = np.where(dim.acc_iaa < np.floor(num_ann/2), num_ann, 0)
            dim['fl_adj_1'] = np.where(dim.acc_iaa > np.ceil(num_ann/2), num_ann, 0)
            dim['fl_adj_0'] = np.where(dim.acc_iaa < np.floor(num_ann/2), num_ann, 0)

            dim['fl_adj_1'] = np.where(~(dim[['fl_adj_1', 'fl_adj_0']].any(axis=1)), np.ceil(num_ann/2), dim['fl_adj_1'])
            dim['fl_adj_0'] = np.where(dim['fl_adj_1'] == np.ceil(num_ann/2), np.floor(num_ann/2), dim['fl_adj_0'])
            fl_adj = fleiss_kappa(dim[['fl_adj_1', 'fl_adj_0']].to_numpy())
            fle_adj.append(round(fl_adj, 3))

            fl = fleiss_kappa(agg).round(3)
            sampl.append(k)
            emo.append(e)
            fle.append(fl)

    df_fl = pd.DataFrame({
        'fleiss':fle,
        'fleiss_adjusted':fle_adj,
        'emotion':emo,
        'sample':sampl,
        'accuracy_perfect': accs_perf,
        'accuracy_majority': accs_maj,
        })

    with st.container():
        fl_res_cat = st.selectbox("Choose categories for analysis of results", ['basic emotions', 'generalised emotions', 'all categories', 'binary'])

        if fl_res_cat == 'basic emotions':
            df_fl = df_fl[df_fl.emotion.isin( cols_emo )]
        elif fl_res_cat == 'generalised emotions':
            df_fl = df_fl[df_fl.emotion.isin( cols_emo1 )]
        elif fl_res_cat == 'binary':
            df_fl = df_fl[df_fl.emotion == 'neutral']
            df_fl.emotion = df_fl.emotion.map({'neutral': 'contains_emotion'})
        add_spacelines(1)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Fleiss kappa")
            col1.metric(fl_res_cat, df_fl.groupby(['sample'])['fleiss'].mean().mean().round(3))
            add_spacelines(1)
            st.subheader("Perfect accuracy")
            col1.metric(fl_res_cat, df_fl.groupby(['sample'])['accuracy_perfect'].mean().mean().round(3))

        with col2:
            st.subheader("Fleiss kappa - adjusted")
            col2.metric(fl_res_cat, df_fl.groupby(['sample'])['fleiss_adjusted'].mean().mean().round(3))
            add_spacelines(1)
            st.subheader("Majority accuracy")
            col2.metric(fl_res_cat, df_fl.groupby(['sample'])['accuracy_majority'].mean().mean().round(3))

            #st.write("Perfect accuracy")
            #st.dataframe(df_fl.groupby(['emotion'], as_index=False)['accuracy_perfect'].agg({'acc_perfect_mean':'mean', 'acc_perfect_max':'max'}).round(3))
            #add_spacelines(1)
            #st.write("Majority accuracy")
            #st.dataframe(df_fl.groupby(['emotion'], as_index=False)['accuracy_majority'].agg({'acc_majority_mean':'mean', 'acc_majority_max':'max'}).round(3))
        add_spacelines(2)

        with st.expander('Results table'):
            add_spacelines(1)
            #st.write("Kappa")
            summary_fl = df_fl.groupby(['emotion'], as_index=False)['fleiss'].agg({'fleiss_mean':'mean'}).round(3)# , 'fleiss_max':'max'
            summary_fl_adj = df_fl.groupby(['emotion'], as_index=False)['fleiss_adjusted'].agg({'fleiss_adj_mean':'mean'}).round(3) # , 'fleiss_adj_max':'max'
            summary_fl = summary_fl.merge(summary_fl_adj, on = 'emotion')
            summary_fl_ap = df_fl.groupby(['emotion'], as_index=False)['accuracy_perfect'].agg({'acc_perfect_mean':'mean'}).round(3) # , 'acc_perfect_max':'max'
            summary_fl_mj = df_fl.groupby(['emotion'], as_index=False)['accuracy_majority'].agg({'acc_majority_mean':'mean'}).round(3) # , 'acc_majority_max':'max'
            summary_fl = summary_fl.merge(summary_fl_ap, on = 'emotion')
            summary_fl = summary_fl.merge(summary_fl_mj, on = 'emotion')
            st.dataframe(summary_fl)
            add_spacelines(1)

        if fl_res_cat in ['generalised emotions', 'all categories']:
            with st.expander('Categories mapping'):
                st.write('Generalised emotions')
                cols_emo1_dict = {
                    'positive': ['joy', 'trust'],
                    'negative_1': ['fear', 'sadness', 'anger', 'disgust'],
                    'negative_2': ['anger', 'disgust']
                    }
                st.write(cols_emo1_dict)


elif page_content == 'Wordclouds':
    cols_emo2_agg = ['joy_agree', 'fear_agree', 'anger_agree', 'sadness_agree', 'disgust_agree', 'trust_agree',
                    'positive_agree', 'negative_1_agree', 'negative_2_agree', 'neutral_agree', 'contains_emotion_agree']

    cols_emo_agg = ['joy_agree', 'fear_agree', 'anger_agree', 'sadness_agree', 'disgust_agree', 'trust_agree']
    cols_emo1_agg = ['positive_agree', 'negative_1_agree', 'negative_2_agree', 'neutral_agree']

    cols_emo2 = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust', 'positive', 'negative_1', 'negative_2', 'neutral', 'contains_emotion']
    agree_cases = df_all.groupby(['id_data'], as_index=False)[cols_emo2].mean().round(3)
    agree_cases.columns = [c+'_agree' for c in agree_cases.columns]
    agree_cases = agree_cases.rename(columns = {'id_data_agree':'id_data'})
    df = df_all.merge(agree_cases, on = 'id_data')
    df = df.drop_duplicates('id_data')
    df = lemmatization(df, 'sentence')

    with st.expander('Categories mapping'):
        cols_emo1_dict = {
            'positive': ['joy', 'trust'],
            'negative_1': ['fear', 'sadness', 'anger', 'disgust'],
            'negative_2': ['anger', 'disgust']
            }
        st.write(cols_emo1_dict)

    add_spacelines(1)
    wcl_cat_agg = st.selectbox("Choose an emotion category for Wordcloud plot", [c.replace('_agree', '') for c in cols_emo2_agg])
    add_spacelines(1)
    wcl_cat_agg = wcl_cat_agg+'_agree'

    cat_iaa_ecl = ['majority voted emotion', 'majority voted NO emotion', 'disaggrement on emotion annotation']
    cat_wcl_iaa = st.radio('Select an IAA category for words in WordCloud', cat_iaa_ecl)
    add_spacelines(1)

    threshold_cloud = st.slider('Select a precision value (threshold) for words in WordCloud', 0, 100, 50)
    st.info(f'Selected value: **{threshold_cloud}**')

    if cat_wcl_iaa == 'majority voted emotion':
        df00 = prepare_cloud_lexeme_data(df[df[wcl_cat_agg] < 0.4], df[df[wcl_cat_agg] > 0.6])
    elif cat_wcl_iaa == 'majority voted NO emotion':
        df00 = prepare_cloud_lexeme_data(df[df[wcl_cat_agg] > 0.6], df[df[wcl_cat_agg] < 0.4])
    elif cat_wcl_iaa == 'disaggrement on emotion annotation':
        df00 = prepare_cloud_lexeme_data(df[ (df[wcl_cat_agg] > 0.6) & (df[wcl_cat_agg] < 0.4)], df[ (df[wcl_cat_agg] > 0.4) & (df[wcl_cat_agg] < 0.6)])


    fig, keyspm = wordcloud_lexeme(dataframe = df00, lexeme_threshold = int(threshold_cloud), cmap_wordcloud = 'RdBu')
    wcl1, wcl2, wcl3, = st.columns([1,8,1])
    with wcl2:
        st.pyplot(fig)
        add_spacelines(1)
        st.write("**Table of words**")
        st.dataframe(pd.DataFrame(keyspm, columns = ['word']))

elif page_content == 'Distribution':
    # majority voting and dist
    cols = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust',
        'positive', 'negative_1', 'negative_2', 'contains_emotion', 'neutral']
    dfs2_2 = {}
    for k in list(dfs2.keys()):
        dd = dfs2[k]
        num_ann = len(dd.ann.unique())

        agree_cases = dd.groupby(['id_data'], as_index=False)[cols].sum().round(3)
        agree_cases.columns = [c+'_agree' for c in agree_cases.columns]
        agree_cases = agree_cases.rename(columns = {'id_data_agree':'id_data'})
        if (num_ann % 2) != 0:
            for c in agree_cases.columns[1:]:
                agree_cases[c] = np.where(agree_cases[c] >= np.ceil(num_ann/2), 1, 0)
        else:
            for c in agree_cases.columns[1:-1]:
                agree_cases[c] = np.where(agree_cases[c] >= np.ceil(num_ann/2), 1, 0)
            agree_cases['neutral_agree'] = np.where(agree_cases['neutral_agree'] > np.ceil(num_ann/2), 1, 0)
        dfs2_2[k] = agree_cases

    dfs2_2_merge = {}
    for k in list(dfs2.keys()):
      dd1 = dfs2[k]
      dd2 = dfs2_2[k]
      dd = dd1.merge(dd2, on = 'id_data')
      dd = dd.drop_duplicates('id_data')
      dfs2_2_merge[k] = dd

    cols_keep = ['id_data', 'full_text_id', 'conversation_id', 'source', 'full_text',
       'sentence', 'joy_agree', 'fear_agree', 'anger_agree',
       'sadness_agree', 'disgust_agree', 'trust_agree', 'positive_agree',
       'negative_1_agree', 'negative_2_agree', 'neutral_agree', 'contains_emotion_agree']

    df = dfs2_2_merge[list(dfs2_2_merge.keys())[0]][cols_keep].copy()
    for k in list(dfs2_2_merge.keys())[1:]:
        df_1 = dfs2_2_merge[k][cols_keep].copy()
        df = pd.concat([df, df_1], axis=0, ignore_index=True)

    df.columns = [c.replace('_agree', '') for c in df.columns]

    cols_emo_bin = ['neutral', 'contains_emotion']
    cols_emo = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust']
    cols_emo1 = ['positive', 'negative_1', 'negative_2', 'neutral']

    colors_dict = {
    'joy' : '#01AD23', 'anger' : '#FD7E00', 'sadness' : '#010598',
    'fear' : '#000000', 'disgust' : '#6A009B', 'positive' : '#00CF80',
    'negative_2' : '#E80000', 'negative_1' : '#900202', 'trust' : '#FFFB07',
    'neutral': '#626464', 'contains_emotion': '#00B9CF'}

    dis_res_cat = st.selectbox("Choose categories for analysis of results", ['binary', 'generalised emotions', 'basic emotions', 'all categories'])
    if dis_res_cat == 'basic emotions':
        cols_dist = cols_emo
        df_plot = (df[cols_dist].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f.set(ylabel='', xlabel='percentage')
        plt.show()
        dc1, dc2, dc3 = st.columns([1,5,1])
        with dc2:
            st.pyplot(f)

    elif dis_res_cat == 'generalised emotions':
        cols_dist = cols_emo1
        df_plot = (df[cols_dist].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f.set(ylabel='', xlabel='percentage')
        plt.show()
        dc1, dc2, dc3 = st.columns([1,5,1])
        with dc2:
            st.pyplot(f)
            add_spacelines()
            with st.expander('Categories mapping'):
                cols_emo1_dict = {
                    'positive': ['joy', 'trust'],
                    'negative_1': ['fear', 'sadness', 'anger', 'disgust'],
                    'negative_2': ['anger', 'disgust']
                    }
                st.write(cols_emo1_dict)

    elif dis_res_cat == 'binary':
        cols_dist = cols_emo_bin
        df_plot = (df[cols_dist].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f.set(ylabel='', xlabel='percentage')
        plt.show()
        dc1, dc2, dc3 = st.columns([1,5,1])
        with dc2:
            st.pyplot(f)

    else:
        df_plot = (df[cols_emo_bin].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f1 = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f1.set(ylabel='', xlabel='percentage')
        plt.show()

        df_plot = (df[cols_emo1].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f2 = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f2.set(ylabel='', xlabel='percentage')
        plt.show()

        df_plot = (df[cols_emo].describe().round(3)*100).T.reset_index()
        sns.set(font_scale=1.4, style='whitegrid')
        f3 = sns.catplot(kind='bar', data = df_plot.sort_values(by = 'mean', ascending=False), x = 'mean', y = 'index', aspect=1.5, palette=colors_dict)
        f3.set(ylabel='', xlabel='percentage')
        plt.show()

        dc1, dc2, dc3 = st.columns([1,5,1])
        with dc2:
            st.pyplot(f1)
            st.pyplot(f2)
            st.pyplot(f3)
            add_spacelines()
            with st.expander('Categories mapping'):
                cols_emo1_dict = {
                    'positive': ['joy', 'trust'],
                    'negative_1': ['fear', 'sadness', 'anger', 'disgust'],
                    'negative_2': ['anger', 'disgust']
                    }
                st.write(cols_emo1_dict)
    add_spacelines(1)


elif page_content == 'Explore cases':
    cols = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'trust',
        'positive', 'negative_1', 'negative_2', 'contains_emotion', 'neutral']
    add_spacelines(1)

    case_emo = st.selectbox("Choose an emotion category", cols)
    add_spacelines(1)

    cat_iaa_ecl = ['majority voted emotion', 'majority voted NO emotion', 'disaggrement on emotion annotation']
    case_type = st.radio('Select a category of annotators agreement', cat_iaa_ecl)
    add_spacelines(2)

    st.write("##### DataFrame")
    # majority voting
    dfs2_2 = {}
    for k in list(dfs2.keys()):
        dd = dfs2[k]
        num_ann = len(dd.ann.unique())

        agree_cases = dd.groupby(['id_data'], as_index=False)[cols].mean().round(2)
        agree_cases.columns = [c+'_agree' for c in agree_cases.columns]
        agree_cases = agree_cases.rename(columns = {'id_data_agree':'id_data'})
        dfs2_2[k] = agree_cases

    dfs2_2_merge = {}
    for k in list(dfs2.keys()):
      dd1 = dfs2[k]
      dd2 = dfs2_2[k]
      dd = dd1.merge(dd2, on = 'id_data')
      dd = dd.drop_duplicates('id_data')
      dfs2_2_merge[k] = dd

    cols_keep = ['id_data', 'full_text_id', 'conversation_id', 'source', 'full_text',
       'sentence', 'joy_agree', 'fear_agree', 'anger_agree',
       'sadness_agree', 'disgust_agree', 'trust_agree', 'positive_agree',
       'negative_1_agree', 'negative_2_agree', 'neutral_agree', 'contains_emotion_agree']

    df = dfs2_2_merge[list(dfs2_2_merge.keys())[0]][cols_keep].copy()
    for k in list(dfs2_2_merge.keys())[1:]:
        df_1 = dfs2_2_merge[k][cols_keep].copy()
        df = pd.concat([df, df_1], axis=0, ignore_index=True)

    df.columns = [c.replace('_agree', '') for c in df.columns]
    cols_keep2 = ['source', 'sentence', 'joy', 'fear', 'anger', 'sadness', 'disgust', 'trust',
        'positive', 'negative_1', 'negative_2', 'contains_emotion', 'neutral']

    if case_type == 'majority voted emotion':
        st.write(f"There are {len(df[ df[case_emo] >= 0.6 ])} such cases. ")
        add_spacelines(1)
        st.dataframe(df[ df[case_emo] >= 0.6 ][cols_keep2])

    elif case_type == 'majority voted NO emotion':
        st.write(f"There are {len(df[ df[case_emo] <= 0.4 ])} such cases. ")
        add_spacelines(1)
        st.dataframe(df[ df[case_emo] <= 0.4 ][cols_keep2])

    elif case_type == 'disaggrement on emotion annotation':
        st.write(f"There are {len( df[ (df[case_emo] < 0.6) & (df[case_emo] > 0.4) ] )} such cases. ")
        add_spacelines(1)
        st.dataframe(df[ (df[case_emo] < 0.6) & (df[case_emo] > 0.4) ][cols_keep2])

    with st.expander('Categories mapping'):
        st.write('Generalised emotions')
        cols_emo1_dict = {
            'positive': ['joy', 'trust'],
            'negative_1': ['fear', 'sadness', 'anger', 'disgust'],
            'negative_2': ['anger', 'disgust']
            }
        st.write(cols_emo1_dict)
