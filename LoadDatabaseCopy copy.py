
import numpy as np
import pandas as pd   
import os
from pathlib import Path
import glob
import json

import os
import nltk
import argparse

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


 
def add_new_row(passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1, acuracy_questao_2, acuracy_questao_3, acuracy_questao_4, porcentagem):
    try:
       query = "INSERT INTO Curriculos VALUES(%s,%s,'%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" %( passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1, acuracy_questao_2, acuracy_questao_3, acuracy_questao_4, porcentagem)
       db.execute(query) 
    except:
        print('erro -- ao executar')
        print(query)
parser = argparse.ArgumentParser()
#parser.add_argument('--file', type=str, default='default')
#config = parser.parse_args()

#file_to_search = config.file
file_to_search = "C:\\source\\PesquisaMestrado\\train_dir\\TreinamentoFaciltudo_novo1\\"   
Files = [f for f in os.listdir(file_to_search) ]
Files.sort()
posQues =[]
valList=[]
trainList=[]
memorys = []
resultpredication = []
Files.sort()
for File in Files:
            
            if 'Logs2.json' in File:

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
                           
               
                           
  

