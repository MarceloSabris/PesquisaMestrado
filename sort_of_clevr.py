from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import h5py

from util import log

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, path, name='default',
                 max_examples=None, is_train=True,batchsize=50):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.batch = []
        self.batchsize = batchsize
        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = 'data.hy'

        file = os.path.join(path, filename)
        log.info("Reading %s ...", file)

        try:
            self.data = h5py.File(file, 'r')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)
        self.maxGrups = int(len(self.ids) / batchsize)
        #self.maxGrups =10
        count = 1
        while(count<self.maxGrups+1):
           self.batch.append(self.get_split(count,batchsize))
           count = count+1

    def get_data(self, id):
        teste=[]
        teste.append(1)
        # preprocessing and data augmentation
        img =teste
        q = self.data[id]['question'][()].astype(np.float32)
        a = self.data[id]['answer'][()].astype(np.float32)
        try: 
           imgDecod =teste
        except: 
           imgDecod = []
        try: 
           codImag = self.data[id]['codImag'][()].astype(np.float32)   
        except: 
           codImag = []
        try:                           
           codImagOri = self.data[id]['codImagOrig'][()].astype(np.float32)   
        except: 
           codImagOri = []
        return img, q, a, imgDecod , codImag,codImagOri
   
    def get_split(self, step,batchsize):
        img=[]
        q=[]
        ids=[]
        a=[]
        imgDecod=[]
        codImag =[]
        codImagOri=[]

        ult = batchsize*step
        inicio = ult - batchsize  
        if (ult > len(self._ids)): 
            stepunicos = int(len(len(self._ids))/batchsize)
            stepunicos = int(step/stepunicos)
            ult = ult -  stepunicos
            inicio = ult - batchsize 
        while(inicio < ult):

          
            id1 = str(inicio)
            ids.append(id1)
            # preprocessing and data augmentation
            img.append([1])
            q.append(self.data[id1]['question'][()].astype(np.float32))
            a.append(self.data[id1]['answer'][()].astype(np.float32))
            try: 
                imgDecod.append([1])   
            except: 
                imgDecod = []
            try: 
                codImag.append(self.data[id1]['codImag'][()].astype(np.float32))
            except: 
                codImag = []
            try:                           
                codImagOri.append(self.data[id1]['codImagOrig'][()].astype(np.float32))   
            except: 
                codImagOri = []
            inicio = inicio +1 
           
        return   [ids,img,q, a, imgDecod , codImag,codImagOri]

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def get_data_info():
    return np.array([128, 128, 3, 11, 10])


def get_conv_info():
    return np.array([24, 24, 24, 24])


def create_default_splits(path, is_train=True, is_shuffe = False,is_full =False,id_filename='id.txt'):
    ids = all_ids(path,is_shuffe,id_filename)
    n = len(ids)

    num_trains = int(n*0.8)
    if (is_full ): 
        num_trains = n
        return  Dataset(ids[:num_trains], path, name='train', is_train=False)
    dataset_train = Dataset(ids[:num_trains], path, name='train', is_train=False)
    dataset_test = Dataset(ids[num_trains:], path, name='test', is_train=False)
    return dataset_train, dataset_test

def return_dataset(path):
    ids = all_ids(path)
    n = len(ids)
    return Dataset(ids[:n], path, name='full', is_train=False)
  

def all_ids(path,is_shuffe=True,id_filename = 'id.txt'):
   
    id_txt = os.path.join(path, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was generated.')
    if (is_shuffe):
        rs.shuffle(_ids)
    return _ids
