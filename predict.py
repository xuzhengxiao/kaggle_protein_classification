from sklearn.metrics import f1_score
import scipy.optimize as opt
import keras
import numpy as np
from resnet import Resnet
import utils
import os
import pandas as pd
import torch
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import f1_score

def sigmoid_np(x):
  return 1.0 / (1.0 + np.exp(-x))

def F1_soft(preds, targs, th=0.5, d=50.0):
  preds = sigmoid_np(d * (preds - th))
  targs = targs.astype(np.float)
  score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
  return score


def fit_val(x, y):
  params = 0.5 * np.ones(len(utils.name_label_dict))
  wd = 1e-5
  error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                    wd * (p - 0.5)), axis=None)
  p, success = opt.leastsq(error, params)
  return p


# use to choose threshold
def generateValData(fnames,is_train=True):
  labels = pd.read_csv(utils.LABELS).set_index('Id')
  labels['Target'] = [[int(i) for i in s.split()] for s in labels['Target']]
  val_data=[]
  val_label=[]
  for i in range(len(fnames)):
    filename=fnames[i]
    img=utils.open_rgby(utils.TRAIN,filename,is_train,utils.stats)
    val_data.append(img)
    label = labels.loc[filename]['Target']
    label = np.eye(len(utils.name_label_dict))[label].sum(axis=0)
    val_label.append(label)

  return np.array(val_data),np.array(val_label)

def process(model,fnames,is_train=False):
  model.cuda()
  model.eval()
  labels = pd.read_csv(utils.LABELS).set_index('Id')
  labels['Target'] = [[int(i) for i in s.split()] for s in labels['Target']]
  val_pred = []
  val_true = []
  for i in range(len(fnames)):
    filename = fnames[i]
    img = utils.open_rgby(utils.TRAIN, filename, is_train, utils.stats)
    img = img[None, ...].astype(dtype=np.float32)
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    img = img.cuda()
    input = Variable(img)
    pred = model(input)
    pred = pred.sigmoid().cpu().data.numpy()
    val_pred.append(pred[0])
    label = labels.loc[filename]['Target']
    label = np.eye(len(utils.name_label_dict))[label].sum(axis=0)
    val_true.append(label)
  return np.array(val_pred),np.array(val_true)

def find(val_pred,val_true):
  ths=np.zeros(28)
  for i in range(28):
    best = -1
    for th in np.arange(0.01,0.99,0.01):
      f1 = f1_score(val_true[:,i], val_pred[:,i] >th , average='macro')
      if f1>best:
        best=f1
        ths[i]=th
  return ths

def choose_th(model):
  val_set = utils.val_n
  val_pred, val_true=process(model,val_set)

  th=fit_val(val_pred,val_true)
  th[th < 0.1] = 0.1
  th[th>0.9]=0.9
  #th=find(val_pred, val_true)
  print('Thresholds: ', th)
  return th


def handle(model,th=0.5,fname='protein_classification.csv'):
  model.cuda()
  model.eval()
  test_set = sorted({f[:36] for f in os.listdir(utils.TEST)})
  pred_list=[]
  preds_t=[]
  for i in range(len(test_set)):
    print (i,' out of ',len(test_set))
    # if i==50:
    #  break
    filename = test_set[i]
    img = utils.open_rgby(utils.TEST, filename,False,utils.stats)
    img=img[None,...].astype(dtype=np.float32)
    #pred=model.predict(img)
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    img = img.cuda()
    input= Variable(img)
    pred = model(input)
    pred = pred.sigmoid().cpu().data.numpy()
    preds_t.append(pred[0])
  preds_t=np.array(preds_t)
  for line in preds_t:
    s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
    pred_list.append(s)
  df = pd.DataFrame({'Id': test_set, 'Predicted': pred_list})
  df.sort_values(by='Id').to_csv(fname, header=True, index=False)


def predict():
  model = Resnet()
  # original saved file with DataParallel
  state_dict = torch.load('sgdr_rgb.pkl')
  # create new OrderedDict that does not contain `module.`
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
  # load params
  model.load_state_dict(new_state_dict)
  #model.load_state_dict(torch.load('best.pkl'))
  th=choose_th(model)
  handle(model,th)

if __name__ == '__main__':
  predict()
