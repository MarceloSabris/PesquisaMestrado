from dataclasses import dataclass
import csv

import click
import requests
import psycopg2
import psycopg2.extras

@dataclass
class Investment:
    id: int
    coin: str
    currency: str
    amount: float

def get_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="manager",
        user="postgres",
        password="pgpassword"
    )

    return connection

@click.group()
def cli():
    pass

@click.command()
@click.option("--coin", prompt=True)
@click.option("--currency", prompt=True)
@click.option("--amount", prompt=True)
def new_investment(coin, currency, amount):
    stmt = f"""
        insert into investment (
            coin, currency, amount
        ) values (
            '{coin.lower()}', '{currency.lower()}', {amount}
        )
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(stmt)
    connection.commit()

    cursor.close()
    connection.close()
    
    print(f"Added investment for {amount} {coin} in {currency}")

@click.command()
@click.option("--filename")
def import_investments(filename):
    stmt = "insert into investment (coin, currency, amount) values %s"

    connection = get_connection()
    cursor = connection.cursor()
    
    with open(filename, 'r') as f:
        coin_reader = csv.reader(f)
        rows = [[x.lower() for x in row[1:]] for row in coin_reader]

    psycopg2.extras.execute_values(cursor, stmt, rows)
    connection.commit()

    cursor.close()
    connection.close()

    print(f"Added {len(rows)} investments")

@click.command()
@click.option("--currency",prompt=True)
def view_investments(currency):
    connection = get_connection()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    stmt = "select * from investment"

    if currency is not None:
        stmt += f" where currency='{currency.lower()}'"

    cursor.execute(stmt)
    data = [Investment(**dict(row)) for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    coins = set([row.coin for row in data])
    currencies = set([row.currency for row in data])

    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(coins)}&vs_currencies={','.join(currencies)}"
    coin_data = requests.get(url).json()

    for investment in data:
        coin_price = coin_data[investment.coin][investment.currency.lower()]
        coin_total = investment.amount * coin_price
        print(f"{investment.amount} {investment.coin} in {investment.currency} is worth {coin_total}")


#cli.add_command(new_investment)
#cli.add_command(import_investments)
cli.add_command(view_investments)

if __name__ == "__main__":
    cli()
