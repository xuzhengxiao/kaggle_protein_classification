from queue import Queue
from threading import Thread
import random
import utils
import pandas as pd
import numpy as np
import torch
import inspect
import ctypes


class Dataset(object):
  def __init__(self,batch_size,thread_num,dirpath,fnames):

    self.threads=[]

    self.batch_size = batch_size
    self.thread_num = thread_num
    self.dirpath=dirpath
    # train data labels
    self.labels = pd.read_csv(utils.LABELS).set_index('Id')
    self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]

    # record and image_label queue
    self.record_queue = Queue(maxsize=5000)
    self.image_label_queue = Queue(maxsize=1024)
    self.batch_queue = Queue(maxsize=32)

    self.record_list = fnames
    self.record_point = 0
    self.record_number = len(self.record_list)

    t_record_producer = Thread(target=self.record_producer)
    t_record_producer.daemon = True
    t_record_producer.start()
    self.threads.append(t_record_producer)

    for i in range(self.thread_num):
      t = Thread(target=self.record_customer)
      t.daemon = True
      t.start()
      self.threads.append(t)

    for i in range(self.thread_num):
      t = Thread(target=self.gen_batch)
      t.daemon = True
      t.start()
      self.threads.append(t)

  def record_producer(self):
    """record_queue's processor
    """
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list)
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def record_process(self, filename,dirpath,is_train,stats):
    img = utils.open_rgby(dirpath, filename, is_train, stats)
    label = self.labels.loc[filename]['Target']
    label = np.eye(len(utils.name_label_dict))[label].sum(axis=0)
    return [img,label]


  def record_customer(self):
    """record queue's customer
    """
    while True:
      item = self.record_queue.get()
      out = self.record_process(item,self.dirpath,is_train=True,stats=utils.stats)
      self.image_label_queue.put(out)


  # generate train and/or validation data
  def gen_batch(self):
    while True:
      images = []
      labels = []
      for i in range(self.batch_size):
        image, label = self.image_label_queue.get()
        images.append(image)
        labels.append(label)

      images = np.asarray(images, dtype=np.float32)
      labels = np.asarray(labels, dtype=np.int32)
      images = torch.from_numpy(images.transpose((0, 3, 1, 2)))
      labels = torch.from_numpy(labels).float()
      self.batch_queue.put((images,labels))

  def batch(self):
    while True:
      images,labels=self.batch_queue.get()
      return (images,labels)

  def _async_raise(self,tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
      exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
      raise ValueError("invalid thread id")
    elif res != 1:
      # """if it returns a number greater than one, you're in trouble,
      # and you should call it again with exc=NULL to revert the effect"""
      ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
      raise SystemError("PyThreadState_SetAsyncExc failed")

  def stop_thread(self,thread):
    self._async_raise(thread.ident, SystemExit)

  def release(self):
    for th in self.threads:
      self.stop_thread(th)
    while not self.record_queue.empty():
      self.record_queue.get()
    while not self.image_label_queue.empty():
      self.image_label_queue.get()
    while not self.batch_queue.empty():
      self.batch_queue.get()