import sqlite3
try: # only use mariadb module if installed
    import mariadb
except ImportError:
    pass


def get_database_connection(config_database_keys, database_name, uri=False):
    '''Return a database connection object, initialized with the given config'''
    database_type = config_database_keys['TYPE']

    db_conn = None
    if database_type == 'sqlite':
        db_conn = sqlite3.connect(database_name, uri=uri)
    elif 'mariadb':
        db_conn = mariadb.connect(
            user=config_database_keys['USER'],
            password=config_database_keys['PASSWORD'],
            host=config_database_keys['HOST'],
            port=config_database_keys['PORT'],
            database=database_name
        )
    else:
        raise(Exception('Invalid database type %s given' % (database_type)))
    return db_conn