from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os.path
import numpy as np
import h5py

from util import log

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, path, name='default',
                 max_examples=None, is_train=True,batchsize=100,is_loadImage=False):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.batch = []
        self.batchsize = batchsize
        self.is_loadImage = is_loadImage
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
    def updateIDS(self,ids):
        self._ids = list(ids)
        self.maxGrups = int(len(self.ids) /  self.batchsize)
        #self.maxGrups =10
        count = 1
        self.batch=[]
        while(count<self.maxGrups+1):
           self.batch.append(self.get_split(count,self.batchsize))
           count = count+1

    def get_data(self, id):
        teste=[]
        teste.append(1)
        if self.is_loadImage==False:
           img =teste
        else: 
        # preprocessing and data augmentation
           img = self.data[id]['image'][()]/255.
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

          
            id1 = str(self._ids[inicio])
            ids.append(id1)
            # preprocessing and data augmentation
            if self.is_loadImage==False:
              img.append([1])
            else: 
              img1 = self.data[id1]['image'][()]/255.
              img.append(img1)
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


def create_default_splits(path, is_train=True, is_shuffe = False,is_full =False,id_filename='id.txt',is_loadImage= False):
    ids = all_ids(path,is_shuffe,id_filename)
    n = len(ids)
    rs.shuffle(ids)
    num_trains = int(n*0.8)
    if (is_full ): 
        num_trains = n
        return  Dataset(ids[:num_trains], path, name='train', is_train=False,is_loadImage = is_loadImage)
    dataset_train = Dataset(ids[:num_trains], path, name='train', is_train=False,is_loadImage = is_loadImage)
    dataset_test = Dataset(ids[num_trains:], path, name='test', is_train=False,is_loadImage = is_loadImage)
    return dataset_train, dataset_test

def create_default_splits_perc(path, is_train=True, is_shuffe = False,is_full =False,tipo='[0]',grupoDatasets=[],OrdemDados='E',is_loadImage= False):
    ids=[] 
    #if len(grupoDatasets) ==0 : 
    #   ids = perc_ids(path,ids,"tipo"+tipo+".txt",porcentual)
    contador = 0
    #else :
    for grupoDataset in grupoDatasets.split(','):
            ids=  perc_ids(path,ids,"tipo"+str(contador)+".txt",float(grupoDataset))
            contador = contador +1   
            if len(OrdemDados) > 0 : 
               if OrdemDados =='E' : 
                  np.random.shuffle(ids)  
   
   
    dataset_train = Dataset(ids, path, name='train', is_train=True,is_loadImage = is_loadImage)
    #dataset_test = Dataset(ids[num_trains:], path, name='test', is_train=False)
    return dataset_train

def updateIdsDataSet (path,data,tipo='[0]',grupoDatasets=[]):
    ids=[] 
    #if len(grupoDatasets) ==0 : 
    #   ids = perc_ids(path,ids,"tipo"+tipo+".txt",porcentual)
    contador = 0
    #else :
    for grupoDataset in grupoDatasets.split(','):
            ids=  perc_ids(path,ids,"tipo"+str(contador)+".txt",float(grupoDataset))
            contador = contador +1     
            np.random.shuffle(ids)
    return data.updateIDS(ids)
    

def return_dataset(path,is_loadImage = False):
    ids = all_ids(path)
    n = len(ids)
    return Dataset(ids[:n], path, name='full', is_train=False,is_loadImage = is_loadImage)
  

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

def count_generator(id_filenamereader):
   with open(id_filenamereader,'r') as myfile:
    total_lines = sum(1 for line in myfile)
   return total_lines;


def perc_ids(path,ids,id_filename = 'id.txt',perc=1.0):
   
    id_txt = os.path.join(path, id_filename)
   
    idsAll = all_ids(path,is_shuffe=True,id_filename =id_filename )
    lines = len(idsAll)
    qtdMAx = lines*perc
    count =0
    while count < qtdMAx :
                  ids.append(idsAll[count])
                  count = count +1 
    return ids
   
  
  