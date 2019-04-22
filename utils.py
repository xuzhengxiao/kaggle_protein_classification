import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
import cv2
import transforms



TRAIN = '../data/train/'
TEST = '../data/test/'
LABELS = '../data/train.csv'


np.random.seed(2050)
stats=np.array([[0.08069,0.05258,0.05487,0.08282],[0.13704,0.10145,0.15313,0.13814]])
name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }

#train_names = sorted({f[:36] for f in os.listdir(TRAIN)})

#tr_n, val_n = train_test_split(train_names, test_size=0.2, random_state=2050)
def convert_targets(row):
  row.Target = [int(i) for i in row.Target.split(" ")]
  row.Target = np.eye(28)[row.Target].sum(axis=0)
  return row


def get_split(test_size=0.2):
  data = pd.read_csv(LABELS)
  data = data.apply(convert_targets, axis=1)
  k_fold = IterativeStratification(n_splits=2, order=2,sample_distribution_per_fold=[test_size, 1.0-test_size])
  train_names=list(data['Id'])
  y=data['Target'].values
  y = np.array([ix for ix in y])
  for train_idx,val_idx in k_fold.split(train_names,y):
    train_set = [train_names[i] for i in train_idx]
    val_set = [train_names[i] for i in val_idx]
    break
  return train_set,val_set


# creating duplicates for rare classes in train set
class Oversampling(object):
  def __init__(self, path):
    self.train_labels = pd.read_csv(path).set_index('Id')
    self.train_labels['Target'] = [[int(i) for i in s.split()]
                                   for s in self.train_labels['Target']]
    # set the minimum number of duplicates for each class
    # self.multi = [1, 1, 1, 1, 1, 1, 1, 1,
    #               8, 8, 8, 1, 1, 1, 1, 8,
    #               1, 2, 1, 1, 4, 1, 1, 1,
    #               2, 1, 2, 8]
    self.multi = [1, 1, 1, 1, 1, 1, 1, 1,
                  5, 4, 3, 1, 1, 1, 1, 7,
                  1, 3, 1, 1, 3, 1, 1, 1,
                  3, 1, 2, 5]

  def get(self, image_id):
    labels = self.train_labels.loc[image_id, 'Target'] if image_id \
                                                          in self.train_labels.index else []
    m = 1
    for l in labels:
      if m < self.multi[l]: m = self.multi[l]
    return m

def get_train(tr_n):
  s = Oversampling(LABELS)
  tr = [idx for idx in tr_n for _ in range(s.get(idx))]
  return tr



def open_rgby( path, filename,train=True,stats=None):
  """ a function that reads RGBY image """
  #colors = ['red', 'green', 'blue', 'yellow']
  colors = ['red', 'green', 'blue']
  flags = cv2.IMREAD_GRAYSCALE
  img = [cv2.imread(os.path.join(path, filename + '_' + color + '.png'), flags).astype(np.float32)
         for color in colors]
  img=[cv2.resize(x,(512,512))/255 for x in img]
  img=np.stack(img, axis=-1)
  ##### do not normalize
  stats=None
  if not stats is None:
    m,s=stats
    img=transforms.Normalize(img,m,s)
  if train:
    img=transforms.RandomRotate(img,30)
    img=transforms.RandomDihedral(img)
    img=transforms.RandomLighting(img,0.05,0.05)
  return img

def fill_targets(row):
  row.Target = np.array(row.Target.split(" ")).astype(np.int)
  for num in row.Target:
    name = name_label_dict[int(num)]
    row.loc[name] = 1
  return row

def count_num():
  data = pd.read_csv(LABELS)
  for key in name_label_dict.keys():
    data[name_label_dict[key]] = 0
  data = data.apply(fill_targets, axis=1)
  target_counts = data.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
  label_count=dict()
  for key,value in name_label_dict.items():
    label_count[key]=target_counts[value]
  return label_count


def create_class_weight(mu=0.5):
  labels_dict=count_num()
  total = np.sum(list(labels_dict.values()))
  keys = labels_dict.keys()
  class_weight = dict()
  class_weight_log = dict()

  for key in keys:
    score = total / float(labels_dict[key])
    score_log = math.log(mu * total / float(labels_dict[key]))
    class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
    class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

  return class_weight, class_weight_log
