
import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json
import array as arr
import os
import nltk
import argparse
import cv2
import matplotlib.pyplot as plt
import random


from os.path import isfile, join
from sqlalchemy import create_engine

from sqlalchemy import create_engine
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='default')
parser.add_argument('--percentualtype', type=str, default='default')
parser.add_argument('--fileGenerate', type=str, default='default')
parser.add_argument('--quantityfile', type=str, default='default')


config = parser.parse_args()

file_to_search = config.file
Files = [f for f in os.listdir(file_to_search) ]
Files.sort()
posQues =[]
valList=[]
trainList=[]
memorys = []
resultpredication = []
Files.sort()
for File in Files:
            
            if '.json' in File:

              '''  if 'ques' in File:     
                    with open(FolderSource + '\\' + File) as f:
                            ques = json.load(f)
          
                            for que in ques :
                                valList.append(que)
                elif 'train' in File:
                    with open(FolderSource + '\\' + File) as f:
                            Ftrain = json.load(f)
                            for tr in Ftrain:
                                trainList.append(tr)
               
                el'''
              usadaEm = "Treinamento"
              Acertou = 0
              if 'id_tipo' in File :
                with open(file_to_search + '\\' + File) as f:
                            fresultpredication = json.load(f)
                            s=0
                            QtdTipo = {"0": 0, "1": 0,"2":0,"3":0,"4":0,"5":0}
                            Questoes=[[],[],[],[],[],[]]
                            Arquivo=[[],[],[],[],[],[]]
                            id1 =0 
                            for t in range(int((len(fresultpredication)))):
                                id = fresultpredication[t]['questaoid']
                                if (int(id) != id1):
                                    print ('Erro-ver geração')
                                id1 = id1 +1
                                tipo =fresultpredication[t]['tipo']
                                QtdTipo[tipo] =QtdTipo[tipo] +1 
                                Questoes[int(tipo)].append(id)
                            print('oi')
                            quantity = 0
                            if (int(config.quantityfile) > 0 ): 
                               if quantity> id1 : 
                                   print("Erro - quantity")
                            
                            valores = config.percentualtype.split(',')
                            if len(valores)>0 : 
                                   for i in valores :     
                                    if (int(config.quantityfile) > 0 ):
                                       quantity=  int(config.quantityfile)
                                    else:
                                       quantity = len(Questoes[int(i.split(':')[0])])
                                    if  int(float(i.split(':')[1])*int(quantity)) > len(Questoes[int(i.split(':')[0])]):
                                       quantity = len(Questoes[int(i.split(':')[0])])
                                    else : 
                                       quantity= int(float(i.split(':')[1])*int(quantity)) 
                                    Arquivo[int(i[0])] =  Questoes[int(i.split(':')[0])][0: quantity] 
                            
                            id_file = open(config.fileGenerate, 'w')
                            for valor in Arquivo:
                              if len(valor)> 0: 
                                  for item in valor:
                                      id_file.write(str(item)+'\n')
                            id_file.close()         
                            
                            '''arquivo = folder.replace('.','_')
                            if 'Train' in File : 
                                arquivo  = arquivo + "_Train_" 
                            else: 
                                arquivo  = arquivo + "_Val_" 
                            arquivo =  arquivo + "_" + resultUltimo[len(resultUltimo) -7:len(resultUltimo)] 
                             linha,qtd,coguinitive,tipo,result
                            '''
                           
                            '''for tr in fresultpredication:
                                add_new_row(i,perc,split[2], split[1],tr)
                                i=i+1'''
                           
               
                           
  

