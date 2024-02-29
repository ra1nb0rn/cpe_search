import mariadb
import sqlite3

SERVERLESS_DATABASES = ['sqlite']

def get_database_connection(config_database_keys, database_name, uri=False):
    '''Return a database connection object, initialized with the given config'''
    database_type = config_database_keys['TYPE']

    db_conn = None
    match database_type:
        case 'sqlite':
            db_conn = sqlite3.connect(database_name, uri=uri)
        case 'mariadb':
            try:
                db_conn = mariadb.connect(
                    user=config_database_keys['USER'],
                    password=config_database_keys['PASSWORD'],
                    host=config_database_keys['HOST'],
                    port=config_database_keys['PORT'],
                    database=database_name
                )
            except:
                db_conn = mariadb.connect(
                    user=config_database_keys['USER'],
                    password=config_database_keys['PASSWORD'],
                    host=config_database_keys['HOST'],
                    port=config_database_keys['PORT']
                )
    return db_conn