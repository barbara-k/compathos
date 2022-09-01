maps_may_pl = glob.glob("/content/drive/MyDrive/Colab Notebooks/debates/debateTVP/*.json")
for i, map in enumerate(maps_may_pl):
  maps_may_pl[i] = map[-17:-5]

maps_june_pl = glob.glob("/content/drive/MyDrive/Colab Notebooks/debates/debateTVN/*.json")
for i, map in enumerate(maps_june_pl):
  maps_june_pl[i] = map[-17:-5]
    
maps_D = glob.glob("/content/drive/MyDrive/Colab Notebooks/debates/US2016D1/*.json")
for i, map in enumerate(maps_D):
  maps_D[i] = map[-17:-5]

all_maps_ids =  maps_D + maps_may_pl + maps_june_pl
len(all_maps_ids) # 120

round(len(all_maps_ids)*0.3) # 36 - number of maps annotated by all annotators
  
(120- 36) / 3 # 28 - number of uniq maps for each annotaror



######  sampling   ######

test_map = all_maps_ids.copy()

three_annotate_maps = np.random.choice(test_map, 36, replace=False) # 30% of all maps
three_annotate_maps = list(three_annotate_maps)
test_map = [x for x in test_map if x not in three_annotate_maps]

size = 28 
print(len(test_map))
print(size)

ch1 = np.random.choice(test_map, size, replace=False)
ch1 = list(ch1)
test_map = [x for x in test_map if x not in ch1]

ch2 = np.random.choice(test_map, size, replace=False)
ch2 = list(ch2)
test_map = [x for x in test_map if x not in ch2]

ch3 = test_map.copy()
print(len(ch1), len(ch2), len(ch3), len(three_annotate_maps))


# create df
df = pd.DataFrame()
df["annotator_id"] = " "
df["uniq_maps_ids"] = " "
df["three_annotation_maps_ids"] = " "
df["number_of_uniq_maps"] = " "
df["number_of_3_maps"] = " "
df["number_all_maps_annotate"] = " "

df.loc[0, :] = [0, ch1, three_annotate_maps, size, len(three_annotate_maps), len_all_maps]
df.loc[1, :] = [1, ch2, three_annotate_maps, size, len(three_annotate_maps), len_all_maps]
df.loc[2, :] = [2, ch3, three_annotate_maps, size, len(three_annotate_maps), len_all_maps]


df.to_excel("/content/drive/MyDrive/Colab Notebooks/debates/sampled_map_ids.xlsx")

