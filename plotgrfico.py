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


#@click.command()
#@click.option("--curriculo",prompt=True)
def view_curriculos(curriculo):
    connection = get_connection()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
           
    stmt = "select  id,passo,curriculo,accuracy_treinamento,accuracy_teste,acuracy_questao_0,acuracy_questao_1,acuracy_questao_2,acuracy_questao_3,acuracy_questao_4, porcentagem from \"Curriculos\""

    if curriculo is not None:
        stmt += f" where curriculo='{curriculo}'"

    cursor.execute(stmt)
    data = [Acao(**dict(row)) for row in cursor.fetchall()]
     
    cursor.close()
    connection.close()



    for acao in data:
     
        print(f"{data.passo}")

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
    view_investmentpand("testedqn5cenarios_04112023_0755_epsod_")





