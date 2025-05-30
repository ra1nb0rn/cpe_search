import os
import re
import sqlite3

try:  # only use mariadb module if installed
    import mariadb
except ImportError:
    pass

SAFE_DBNAME_REGEX_MARIADB = re.compile(r"^[\w\-]*$")
SAFE_DBNAME_REGEX_SQLITE = re.compile(r"^[\w\-\. /]*$")
CONNECTION_POOL_SIZE = (
    os.cpu_count()
)  # should be equal to number of cpu cores? (https://dba.stackexchange.com/a/305726)
CONNECTION_POOLS = {}


def get_database_connection(database_config, db_name=None, use_pool=True, sqlite_timeout=None):
    """Return a database connection object, initialized with the given config"""
    database_type = database_config["TYPE"]

    if db_name is None:
        db_name = database_config.get("NAME", "")

    db_conn = None
    if database_type == "sqlite":
        if sqlite_timeout:
            db_conn = sqlite3.connect(db_name, timeout=sqlite_timeout)
        else:
            db_conn = sqlite3.connect(db_name)
    elif database_type == "mariadb":
        # try to use connection pools
        pool_name = "pool_" + db_name
        if pool_name in CONNECTION_POOLS:
            try:
                db_conn = CONNECTION_POOLS[pool_name].get_connection()
            except:
                # no connection in pool available
                db_conn = mariadb.connect(
                    user=database_config["USER"],
                    password=database_config["PASSWORD"],
                    host=database_config["HOST"],
                    port=database_config["PORT"],
                    database=db_name,
                )
        elif use_pool:
            conn_params = {
                "user": database_config["USER"],
                "password": database_config["PASSWORD"],
                "host": database_config["HOST"],
                "port": database_config["PORT"],
                "database": db_name,
            }
            pool = mariadb.ConnectionPool(
                pool_name=pool_name, pool_size=CONNECTION_POOL_SIZE, **conn_params
            )
            CONNECTION_POOLS[pool_name] = pool
            return get_database_connection(database_config, db_name=db_name)
        else:
            db_conn = mariadb.connect(
                user=database_config["USER"],
                password=database_config["PASSWORD"],
                host=database_config["HOST"],
                port=database_config["PORT"],
                database=db_name,
            )
    else:
        raise (Exception("Invalid database type %s given" % (database_type)))
    return db_conn


def get_connection_pools():
    return CONNECTION_POOLS


def is_safe_db_name(db_name, db_type="sqlite"):
    if db_type == "sqlite":
        return SAFE_DBNAME_REGEX_SQLITE.match(db_name)
    else:
        return SAFE_DBNAME_REGEX_MARIADB.match(db_name)
