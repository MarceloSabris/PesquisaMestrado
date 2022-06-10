
import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json

import os
import nltk
import argparse
import cv2
import matplotlib.pyplot as plt
import random


from os.path import isfile, join
from sqlalchemy import create_engine

from sqlalchemy import create_engine

db_name = 'postgres'
db_user = 'postgres'
db_pass = 'mudar123'
db_host = 'localhost'
db_port = '5432'

db_string = 'postgres://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)

 
def add_new_row(questaoid,ordem,pergunta,Acertou,resposta,Tipo,usadaEm,porcentual,pasta,curriculum,coguinitividade,RelacionalNaoRelacional):
    try:
       query = "INSERT INTO mestrado.processamento2 (questaoid,ordem,pergunta,Acertou,resposta,Tipo,usadaEm,porcentual,pasta,curriculum,coguinitividade,relacional_nao_relacional) VALUES(%s,%s,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" %( questaoid,ordem,pergunta,Acertou,resposta,Tipo,usadaEm,porcentual,pasta,curriculum,coguinitividade,RelacionalNaoRelacional)
       db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='default')
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
              if 'Treinamento' in File :
                  usadaEm = "Treinamento"
              if 'Teste' in File :
                  usadaEm = "Teste"
              if 'Certa' in File :
                  Acertou = 1
              if 'Errada' in File :
                  Acertou = 0
              
              with open(file_to_search + '\\' + File) as f:
                            fresultpredication = json.load(f)
                            s=0
                            for t in range(int((len(fresultpredication)/6))):
                                questaoid = int(fresultpredication[s].split(':')[1])
                                s =s+1 
                                pergunta = fresultpredication[s]
                                
                                s=s+1 
                                resposta = fresultpredication[s] + "-" + fresultpredication[s+1]
                                s = s+2
                                Tipo = int(fresultpredication[s].split(":")[1]) 
                                s = s+1
                                
                                porcentual = file_to_search.split("\\")[len(file_to_search.split("\\"))-1]
                                pasta = file_to_search
                                curriculum="1"
                                RelacionalNaoRelacional = fresultpredication[s].split(":")[1]
                                coguinitividade=porcentual.split('-')[1]
                                s=s+1
                                add_new_row(questaoid,questaoid,pergunta,Acertou,resposta,Tipo,usadaEm,porcentual,pasta,curriculum,coguinitividade,RelacionalNaoRelacional)
                              
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
                           
               
                           
  

