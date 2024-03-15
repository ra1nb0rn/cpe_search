import os
import sqlite3

try: # only use mariadb module if installed
    import mariadb
except ImportError:
    pass

CONNECTION_POOL_SIZE = os.cpu_count() # should be equal to number of cpu cores? (https://dba.stackexchange.com/a/305726)
CONNECTION_POOLS = {}


def get_database_connection(config_database_keys, database_name):
    '''Return a database connection object, initialized with the given config'''
    database_type = config_database_keys['TYPE']

    db_conn = None
    if database_type == 'sqlite':
        db_conn = sqlite3.connect(database_name)
    elif database_type == 'mariadb':
        # try to use connection pools
        pool_name = 'pool_' + database_name
        if pool_name in CONNECTION_POOLS:
            try:
                db_conn = CONNECTION_POOLS[pool_name].get_connection()
            except:
                # no connection in pool available
                db_conn = mariadb.connect(
                    user=config_database_keys['USER'],
                    password=config_database_keys['PASSWORD'],
                    host=config_database_keys['HOST'],
                    port=config_database_keys['PORT'],
                    database=database_name
                )
        else:
            conn_params = {
                'user': config_database_keys['USER'],
                'password': config_database_keys['PASSWORD'],
                'host': config_database_keys['HOST'],
                'port': config_database_keys['PORT'],
                'database': database_name
            }
            pool = mariadb.ConnectionPool(pool_name=pool_name, pool_size=CONNECTION_POOL_SIZE, **conn_params)
            CONNECTION_POOLS[pool_name] = pool
            return get_database_connection(config_database_keys, database_name)
    else:
        raise(Exception('Invalid database type %s given' % (database_type)))
    return db_conn


def get_connection_pools():
    return CONNECTION_POOLS
