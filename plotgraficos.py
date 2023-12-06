from dataclasses import dataclass
import csv

import click
import requests
import psycopg2
import psycopg2.extras
import pandas.io.sql as sqlio
import ipympl 
import os

# importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

@dataclass
class Acao:
    id: int
    passo: int
    curriculo: str
    accuracy_treinamento: float
    accuracy_teste : float
    acuracy_questao_0: float 
    acuracy_questao_1: float
    acuracy_questao_2: float 
    acuracy_questao_3: float
    acuracy_questao_4: float
    porcentagem : str   
    

def get_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="senha"
    )

    return connection

@click.group()
def cli():
    pass


def remove(epsodio):
  return epsodio[-2:].replace("_","")

def curriculos_geral(caminho,curriculo):
    connection = get_connection()
    stmt = f"select passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1,acuracy_questao_2,acuracy_questao_3,acuracy_questao_4,porcentagem FROM \"Curriculos\" where curriculo like '{curriculo}_%' order by cast(replace(SUBSTR (curriculo,LENGTH(curriculo)-1,2 ),'_','') as INTEGER), passo" 
    data = sqlio.read_sql_query(stmt, connection)
    data1 = data 
    data1['ep'] = data['curriculo'].apply(remove)
    data1 = data1.drop('curriculo', axis=1)
    data1 = data1.drop('accuracy_treinamento', axis=1)
    data1 = data1.drop('accuracy_teste', axis=1)
    data1 = data1.drop('porcentagem', axis=1)
    data1 = data1.astype({'ep':'int'}) 
    data1 = data1.astype({'acuracy_questao_0':'float'}) 
    data1 =data1.astype({'acuracy_questao_1':'float'})
    data1 =data1.astype({'acuracy_questao_2':'float'})
    data1 =data1.astype({'acuracy_questao_3':'float'})
    data1 =data1.astype({'acuracy_questao_4':'float'})
    teste = data1['ep'].unique()
    my_list_quest0 = []
    my_list_quest1 = []
    my_list_quest2 = []
    my_list_quest3 = []
    my_list_quest4 = []
    passo_list =[]
        
    for i in teste: 
        dataEp0 = data1.loc[data1['ep'] ==i ]
        passo_list.append(dataEp0['passo'])
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.plot(dataEp0['passo'], dataEp0['acuracy_questao_0'],label="Quest-0")
        ax.plot(dataEp0['passo'], dataEp0['acuracy_questao_1'],label="Quest-1")
        ax.plot(dataEp0['passo'], dataEp0['acuracy_questao_2'],label="Quest-2")
        ax.plot(dataEp0['passo'], dataEp0['acuracy_questao_3'],label="Quest-3")
        ax.plot(dataEp0['passo'], dataEp0['acuracy_questao_4'],label="Quest-4")
        plt.style.use('fivethirtyeight')
        #fig.legend(loc='upper center', bbox_to_anchor=(1, .4))
        ax.legend(loc='upper center',ncol=5,title="Questões X Acuracy Ep" + str(i) ,bbox_to_anchor=(0.5, 1.2))
        fig.autofmt_xdate() 
        fig.savefig('train_dir/' + caminho + '/graficos/acuracyquestoes_ep'+str(i)+'.png')
        plt.close(fig)
        my_list_quest0.append(dataEp0['acuracy_questao_0'])
        my_list_quest1.append(dataEp0['acuracy_questao_1'])
        my_list_quest2.append(dataEp0['acuracy_questao_2'])
        my_list_quest3.append(dataEp0['acuracy_questao_3'])
        my_list_quest4.append(dataEp0['acuracy_questao_4'])
    fig, ax = plt.subplots(figsize = (15, 9))
    for i in teste:
        passo = passo_list[i]
        ax.plot(passo_list[i], my_list_quest0[i],label="EP" + str(i))
    ax.legend(loc='upper center',ncol=8,title="Epsodio x Questão 0",bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig('train_dir/' + caminho + '/graficos/acuracyQuestoes0PorEpso.png')
    plt.close(fig)   
    fig, ax = plt.subplots(figsize = (15, 9))
    for i in teste:
        ax.plot(passo_list[i], my_list_quest1[i],label="EP" + str(i))
    ax.legend(loc='upper center',ncol=8,title="Epsodio x Qiestão 1",bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig('train_dir/' + caminho + '/graficos/acuracyQuestoes1PorEpso.png')
    plt.close(fig)   
    fig, ax = plt.subplots(figsize = (15, 9))
    for i in teste:
        ax.plot(passo_list[i], my_list_quest2[i],label="EP" + str(i))
    ax.legend(loc='upper center',ncol=8,title="Epsodio x Questão 2",bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig('train_dir/' + caminho + '/graficos/acuracyQuestoes2PorEpso.png')
    plt.close(fig)   
    fig, ax = plt.subplots(figsize = (15, 9)) 
    for i in teste:
        ax.plot(passo_list[i], my_list_quest3[i],label="EP" + str(i))
    fig.autofmt_xdate()
    ax.legend(loc='upper center',ncol=8,title="Epsodio x Questão 3",bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig('train_dir/' + caminho + '/graficos/acuracyQuestoes3PorEpso.png')
    plt.close(fig)   
    fig, ax = plt.subplots(figsize = (15, 9)) 
    for i in teste:
        ax.plot(passo_list[i], my_list_quest4[i],label="EP" + str(i))
    ax.legend(loc='upper center',ncol=8,title="Epsodio x Questao 4",bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig('train_dir/' + caminho + '/graficos/acuracyQuestoes4PorEpso.png')
    plt.close(fig)   
                 
#ax.set_title('Yellow Taxi Trips in New York City')
#ax.set_xlabel('Pickup Date')
#ax.set_ylabel('Trips')
#ax.set_ylim(0, 120000)
    
    

#@click.command()
#@click.option("--curriculo",prompt=True)
def curriculos_total(caminho,curriculo):
    connection = get_connection()
    
           
    stmt = f"SELECT replace(SUBSTR (curriculo,LENGTH(curriculo)-7,9 ),'_e','e') epsodio,  porcentagem curriculo, count(porcentagem) qtd FROM \"Curriculos\" where curriculo like '{curriculo}_%' GROUP BY curriculo,porcentagem   order by cast(replace(SUBSTR (curriculo,LENGTH(curriculo)-1,2 ),'_','') as INTEGER), porcentagem" 

    data = sqlio.read_sql_query(stmt, connection)
    
    
    data = sqlio.read_sql_query(stmt, connection)
     
    

    data3 = data 
    data3['ep'] = data['epsodio'].apply(remove)
    data3 = data3.drop('epsodio', axis=1)
    data3 = data3.astype({'ep':'int'})
    teste = data3.pivot(columns=['curriculo'],index='ep')
    
    # reverse to keep order consistent
    os.makedirs('train_dir/' + caminho + '/graficos/', exist_ok=True)

     
    fig, ax = plt.subplots(figsize = (15, 9)) 
    ax.legend(loc='upper right',ncol=2,title="curriculos")
    plt.style.use('fivethirtyeight')
    teste.plot(kind="bar",ax=ax ,width = 1,figsize=(9,7),rot=1)
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    # reverse to keep order consistent
    ax.legend(loc='upper right',ncol=8,title="Total de curriculos X Ep")
    fig.savefig('train_dir/' + caminho + '/graficos/totalclporepisodio.png')
    plt.close(fig)
 
   
    grup = data3.drop('ep', axis=1)
    grup = grup.groupby(['curriculo']).sum()
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()

    grup.plot.pie(subplots=True, ax=ax ,figsize=(11, 6))
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    
    fig.savefig('train_dir/' + caminho + '/graficos/totalgeral.png')
    plt.close(fig)



 
    stmt = f"select passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1,acuracy_questao_2,acuracy_questao_3,acuracy_questao_4,porcentagem FROM \"Curriculos\" where curriculo like '{curriculo}_%' order by cast(replace(SUBSTR (curriculo,LENGTH(curriculo)-1,2 ),'_','') as INTEGER), passo" 

    data = sqlio.read_sql_query(stmt, connection)
    data1 = data 
    data1['ep'] = data['curriculo'].apply(remove)
    data1 = data1.drop('curriculo', axis=1)
    data1 = data1.drop('accuracy_treinamento', axis=1)
    data1 = data1.drop('accuracy_teste', axis=1)

    data1 = data1.astype({'ep':'int'}) 

    data1 = data1.astype({'passo':'int'}) 

    teste = data1['ep'].unique()
    for i in teste: 
        dataEp0 = data1.loc[data1['ep'] ==i ]
        fig, ax = plt.subplots(figsize = (15, 9)) 
        ax.legend(loc='upper right',ncol=8,title="Utilização dos currículos por ep"+ str(i))
        ax.plot( dataEp0['porcentagem'],dataEp0['passo'])
        fig.autofmt_xdate() 
        
        fig.savefig('train_dir/' + caminho + '/graficos/gaficoalteracao_ep'+ str(i) +  '.png')
        plt.close(fig)
    connection.close()



    

#@click.command()
#@click.option("--curriculo",prompt=True)
def view_investmentpand(curriculo):
    connection = get_connection()
          
    stmt = f"SELECT replace(SUBSTR (curriculo,LENGTH(curriculo)-7,9 ),'_e','e') epsodio,  porcentagem curriculo, count(porcentagem) qtd FROM \"Curriculos\" where curriculo like '{curriculo}_%' GROUP BY curriculo,porcentagem   order by cast(replace(SUBSTR (curriculo,LENGTH(curriculo)-1,2 ),'_','') as INTEGER) " 

    data = sqlio.read_sql_query(stmt, connection)
    # Now data is a pandas dataframe having the results of above query.
    data.head()
    #plt.bar(data['epsodio'],data['qtd']) 
    


    

#cli.add_command(view_investments)

if __name__ == "__main__":
    #cli()
    curriculos_total("testeNovoSetup3cenarisdqn3cenarios01_23112023_0153_epsod","testeNovoSetup3cenarisdqn3cenarios01_23112023_0153_epsod")
    curriculos_geral("testeNovoSetup3cenarisdqn3cenarios01_23112023_0153_epsod","testeNovoSetup3cenarisdqn3cenarios01_23112023_0153_epsod")
    





