
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
db_pass = 'senha'
db_host = 'localhost'
db_port = '5432'

db_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)
query =""

 
def add_new_row(passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1, acuracy_questao_2, acuracy_questao_3, acuracy_questao_4, porcentagem):
    try:
       import re
       query = "INSERT INTO \"Curriculos\" VALUES(%s,\'%s\',%s,%s,%s,%s,%s,%s,%s,\'%s\')" %( passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1, acuracy_questao_2, acuracy_questao_3, acuracy_questao_4, porcentagem)
       my_new_string = re.sub('\n\.]', '', query)
       db.execute(my_new_string) 
    except:
        print('erro -- ao executar')
        print(query)
