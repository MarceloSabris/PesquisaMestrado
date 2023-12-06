from dataclasses import dataclass
import csv

import click    
import requests
import psycopg2
import psycopg2.extras
import pandas.io.sql as sqlio
import ipympl 


# importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random 
import time 

num_actions = 3
i=100
actionantiga = 10
contaigual = 0 
while i > 0:
   action = random.randint(0,num_actions-1) 
   
   if actionantiga == action: 
      contaigual = contaigual +1
   actionantiga = action   
   i = i-1
print (contaigual)


i=100
repetition = 10
actionantiga = 10
contaigual = 0 
while i > 0:
   random.seed(time.perf_counter())
   action = random.randint(0,num_actions-1) 
   
   if actionantiga == action: 
      contaigual = contaigual +1
   actionantiga = action   
   i = i-1
print (contaigual)
