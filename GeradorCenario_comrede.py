from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import plaidml.keras
import os
#plaidml.keras.install_backend()

import numpy
from six.moves import xrange

import matplotlib.pyplot as plt
from util import log

from pprint import pprint

from input_ops import create_input_ops
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from vqa_util import NUM_COLOR
from vqa_util import visualize_iqa,question2str,answer2str

import time
import tensorflow as tf
import tf_slim as slim
import numpy
import json
import logging
from datetime import datetime
import random
import DataBase as Postgree


tf.get_logger().setLevel(logging.ERROR)
tf.debugging.set_log_device_placement(True) 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#tf.enable_eager_execution() 
class GerarTreinamento(object):

        
    def __init__(self,config):
        print("teste")
        self.config = config
               
    def split_with_sum(self,limit, num_elem, tries=50):
       v = np.random.randint(0, limit, num_elem)
       
       s = sum(v)
       if (np.sum(np.round(v/s*limit)) == limit):
        return np.round(v / s * limit)
       elif (np.sum(np.floor(v/s*limit)) == limit):
        return np.floor(v / s * limit)
       elif (np.sum(np.ceil(v/s*limit)) == limit):
        return np.ceil(v / s * limit)
       else:
        return self.split_with_sum(limit, num_elem, tries-1)
    
    def split_with_zero(self,limit, num_elem, tries=50):
       v = np.random.rand(num_elem)
       s = sum(v)
       if limit == 0 :
        return np.round(v,1)
       elif   (np.sum(np.round(v/s*limit,1)) == limit):
        return np.round(v / s * limit,1)
       
       else:
        return self.split_with_zero(limit, num_elem, tries-1)

    def run(self ): 
      
       for rod in range(self.config.QtdRun):
            print("**********************rodada *********************")
            print(rod)
            #StepChangeGroup = self.split_with_sum( self.config.TotalStepChangeGroup,self.config.QtdStepChangeGroup)
            #orderDataset = [item for item in range(0,self.config.QtdStepChangeGroup )]   
            GrupDataset = ""
            for grup in range(self.config.QtdStepChangeGroup): 
                GrupDataset += ','.join(map(str, self.split_with_zero(self.config.PorcentualGrupDataset,5))) + '|'
            GrupDataset = GrupDataset.rstrip(GrupDataset[-1])
            self.config.train_dir = GrupDataset.replace("|", "_")
            #self.config.StepChangeGroup = ','.join(map(str, StepChangeGroup.astype(int))) 
            self.config.GrupDataset =  GrupDataset 
            #self.config.orderDataset = ','.join(map(str, orderDataset))      
            for test in range(3):
              import os
              teste = ' python d:\\\\source\\PesquisaMestrado\\trainerCodImag.py --StepChangeGroup "' + self.config.StepChangeGroup + '" --orderDataset "'+self.config.orderDataset +'" --GrupDataset "' + self.config.GrupDataset +'" --model "rn" --batch_size "100" --check_pathSaveTrain "d:\\\\train_result_dinamico\\\\" --train_dir "' + self.config.train_dir + '" --is_loadImage False '
                        #python d:\source\PesquisaMestrado\trainerCodImag.py --StepChangeGroup        "25000,75000"                     --orderDataset "0,1"                           --GrupDataset  "0,0,0,1,1|1,1,1,1,1"           --model "rn" --batch_size "100" --check_pathSaveTrain "d:\\train_result_dinamico\\" --train_dir "cenario4_novo2500"             --is_loadImage "False"  

              
              print(teste)
              os.system(teste)


def main():
    

    import tensorflow as tf
    tf.test.is_gpu_available()

    #gpu_config = tf.GPUOptions()
    #gpu_config.visible_device_list = "0"

    #session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))
    import argparse
    #import os
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_teste_decode-image3')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--train_amount_network',  type=int, default=0)
    parser.add_argument('--train_type',  type=str , default='full')
    parser.add_argument('--train_nameDataset',  type=str , default='id')
    
    parser.add_argument('--GrupDataset',  type=str , default='0,1,2|3,4')
    parser.add_argument('--OrdemDados',  type=str , default='E,E')
    parser.add_argument('--check_pathSaveTrain', type=str , default='id')
    parser.add_argument('--train_dir', type=str , default='padrao')
    parser.add_argument('--orderDataset', type=str , default='1,0')
    parser.add_argument('--StepChangeGroup', type=str , default='50000,50000')
    parser.add_argument('--is_loadImage', type=str , default=False)
    parser.add_argument('--TotalStepChangeGroup', type=int, default=10000)
    parser.add_argument('--QtdStepChangeGroup', type=int, default='3')
    parser.add_argument('--QtdRun', type=int, default='3')
    parser.add_argument('--PorcentualGrupDataset', type=int, default='0')
    
    config = parser.parse_args()
    config.is_loadImage = eval(config.is_loadImage)
    path = os.path.join('./datasets', config.dataset_path  )
    #porcentual = config.train_nameInitalDataset.split(',')
    GrupDataset = config.GrupDataset.split('|')
   
    trainer = GerarTreinamento(config)
    trainer.run()

    


   

if __name__ == '__main__':
    main()
