import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

from recipes.steps.ingest.datasets import PolarsDataset

# Connection details as variables
username = 'postgres'
password = 'postgres123'
hostname = '192.168.1.65'
database_name = 'ingest'
port = '5433'  # Default PostgreSQL port
df_url = '/home/khaldi/Documents/github_repos/refined_mlflow/avs/fail_nff/sru_datatset.csv'
sep = ','
encoding = 'ISO-8859-1'


def create_database(conn, database_name):
    # Create a cursor object
    cursor = conn.cursor()
    # Execute the SQL command to create the database
    cursor.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database_name)))

    # Close the cursor and connection
    cursor.close()
    conn.close()


if __name__ == '__main__':
    conn = psycopg2.connect(dbname='postgres', user=username, password=password, host=hostname, port=port)
    conn.autocommit = True  # Enable autocommit mode
    # create_database(conn, database_name)
    # Create the connection string using the variables
    connection_string = f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database_name}'
    # Create the PostgreSQL engine
    engine = create_engine(connection_string)
    # Sample DataFrame
    df = PolarsDataset.read_csv(df_url, separator=sep, encoding=encoding).to_pandas().iloc[:, 1:]
    # Write DataFrame to PostgreSQL table
    df.to_sql('sru_dataset', engine, if_exists='replace', index=False)
    # Dispose of the engine
    engine.dispose()
