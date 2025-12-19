#!/usr/bin/env python3

import argparse
import math
import os
import pprint
import re
import string
import sys
import time
from collections import Counter

import ujson

from cpe_search.database_wrapper_functions import *

# Constants
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CPE_API_URL = "https://services.nvd.nist.gov/rest/json/cpes/2.0/"
DEFAULT_CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
CREATE_SQL_STATEMENTS_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "create_sql_statements.json"
)
TEXT_TO_VECTOR_RE = re.compile(r"[\w+\.]+")
SPLIT_QUERY_TERMS_RE = re.compile(r"[ _\-\.]")
CPE_TERM_WEIGHT_EXP_FACTOR = -0.08
QUERY_TERM_WEIGHT_EXP_FACTOR = -0.25
GET_ALL_CPES_RE = re.compile(r"(.*);.*;.*")
VERSION_MATCH_ZE_RE = re.compile(r"\b([\d]+\.?){1,4}\b")
VERSION_MATCH_CPE_CREATION_RE = re.compile(
    r"\b((\d[\da-zA-Z\.]{0,6})([\+\-\.\_\~ ][\da-zA-Z\.]+){0,4})[^\w\n]*$"
)
VERSION_SPLIT_DIFF_CHARSETS_RE = re.compile(r"(?<=\d)(?=[^\d.])")
MATCH_CPE_23_RE = re.compile(r"cpe:2\.3:[aoh](:[^:]+){2,10}")
CPE_SEARCH_THRESHOLD_ALT = 0.25
TERMS = []
TERMS_MAP = {}
ALT_QUERY_MAXSPLIT = 1
API_CPE_RESULTS_PER_PAGE = 10000
UPDATE_SUCCESS = True
SILENT = True
DEBUG = False
CPE_CREATION_DEL_SYMBOLS_RE = re.compile(r'[\]"\|{>)/`<#},\[\:(=;^\'%]')
POPULAR_QUERY_CORRECTIONS = {
    "flask": "palletsprojects",
    "keycloak": "redhat red hat",
    "rabbitmq": "vmware",
    "bootstrap": "getbootstrap",
    "kotlin": "jetbrains",
    "spring boot": "vmware",
    "debian": "linux",
    "ansible": "redhat",
    "twig": "symfony",
    "proxmox ve": "virtual environment",
    "nextjs": "vercel",
    "next.js": "vercel",
    "ubuntu": "linux",
    "symfony": "sensiolabs",
    "electron": "electronjs",
    "microsoft exchange": "server",
}
QUERY_ABBREVIATIONS = {
    "adc": (["citrix"], "application delivery controller"),
    "omsa": (["dell"], "openmanage server administrator"),
    "cdk": (["amazon", "aws"], "aws cdk cloud development kit"),
    "srm": (["vmware"], "site recovery manager"),
    "paloaltonetworks": ([], "palo alto networks"),
    "palo alto networks": ([], "paloaltonetworks"),
    "trend micro": ([], "trendmicro"),
    "ds": (["trend", "micro"], "deep security"),
    "ms": ([], "microsoft"),
    "dsa": (["trend", "micro"], "deep security agent"),
    "dsm": (["trend", "micro"], "deep security manager"),
    "asa": (["cisco"], "adaptive security appliance"),
}
TF_IDF_DEDUPLICATION_KEYWORDS = {"apache": 1, "flask": 1}


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)"
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update the local CPE database"
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="NVD API key to use for updating the local CPE dictionary",
    )
    parser.add_argument(
        "-n",
        "--number",
        default=3,
        type=int,
        help="The number of CPEs to show in the similarity overview (default: 3)",
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="queries",
        metavar="QUERY",
        action="append",
        help="A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Be verbose and print status information"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="A config file to use (default: config.json)",
    )

    args = parser.parse_args()
    if not args.update and not args.queries:
        parser.print_help()
    return args


def _load_config(config_file=DEFAULT_CONFIG_FILE):
    """Load config from file"""

    def load_config_dict(_dict, parent_key):
        config = {}
        db_type = _dict["type"] if "type" in _dict else _dict.get("TYPE", "")

        for key, val in _dict.items():
            if isinstance(val, dict):
                val = load_config_dict(val, key)
            elif "file" in key.lower() or (
                key.lower() == "name"
                and "database" in parent_key.lower()
                and db_type.lower() == "sqlite"
            ):

                if not os.path.isabs(val):
                    if val != os.path.expanduser(val):  # home-relative path was given
                        val = os.path.expanduser(val)
                    else:
                        val = os.path.join(os.path.dirname(os.path.abspath(config_file)), val)
            config[key] = val
        return config

    with open(config_file) as f:  # default: config.json
        config_raw = ujson.loads(f.read())
        config = load_config_dict(config_raw, "")

    return config


async def api_request(headers, params, requestno):
    """Perform request to API for one task"""

    global UPDATE_SUCCESS

    if not UPDATE_SUCCESS:
        return None

    retry_limit = 3
    retry_interval = 6
    for _ in range(retry_limit + 1):
        async with aiohttp.ClientSession() as session:
            try:
                cpe_api_data_response = await session.get(
                    url=CPE_API_URL, headers=headers, params=params
                )
                if cpe_api_data_response.status == 200:
                    if DEBUG:
                        print(f"[+] Successfully received data from request {requestno}.")
                    return await cpe_api_data_response.json()
                else:
                    if DEBUG:
                        print(
                            f"[-] Received status code {cpe_api_data_response.status} on request {requestno} Retrying..."
                        )
                await asyncio.sleep(retry_interval)
            except Exception as e:
                if UPDATE_SUCCESS and not SILENT:
                    print(
                        "Got the following exception when downloading CPE data via API: %s"
                        % str(e)
                    )
    UPDATE_SUCCESS = False
    return None


def intermediate_process(api_data, requestno):
    """Performs extraction of CPE names and CPE titles"""

    if DEBUG:
        print(f"[+] Intermediate processing on request number {requestno}")

    products = api_data.get("products")
    cpes = []
    deprecations = []
    for product in products:
        extracted_title = ""
        deprecated = product["cpe"].get("deprecated", False)
        cpe_name = product["cpe"]["cpeName"]
        for title in product["cpe"]["titles"]:
            # assume an english title is always present
            if title["lang"] == "en":
                extracted_title = title["title"]
        if deprecated:
            deprecated_by = {cpe_name: []}
            for item in product["cpe"].get("deprecatedBy", []):
                deprecated_by[cpe_name].append(item.get("cpeName"))
            deprecations.append(deprecated_by)
        cpes.append(str(cpe_name + ";" + extracted_title + ";"))
    return cpes, deprecations


def compute_cpe_entry_tf_norm(cpe, name=""):
    """Compute term frequencies and normalization factor for given CPE and name"""
    cpe_mod = cpe.replace("_", ":").replace("*", "").replace("\\", "")
    cpe_name = name.lower()

    cpe_elems = [cpe_part for cpe_part in cpe_mod[10:].split(":") if cpe_part != ""]
    cpe_name_elems = [word for word in cpe_name.split()]

    # compute term weights with exponential decay according to word position
    words_cpe = TEXT_TO_VECTOR_RE.findall(" ".join(cpe_elems))
    words_cpe_name = TEXT_TO_VECTOR_RE.findall(" ".join(cpe_name_elems))

    # deduplicate certain keywords, e.g. for cpe:2.3:a:apache:apache-airflow-providers-apache-hive:
    for keyword, max_count in TF_IDF_DEDUPLICATION_KEYWORDS.items():
        if keyword in words_cpe:
            kw_count = words_cpe.count(keyword)
            del_idxs = []
            for i, word in enumerate(words_cpe):
                # allow keyword to appear 2 times max
                if kw_count < max_count + 1 or kw_count == len(words_cpe):
                    break
                if word == keyword:
                    del_idxs.append(i)
                    kw_count -= 1
            for i, idx in enumerate(del_idxs):
                del words_cpe[idx - i]

    for keyword, max_count in TF_IDF_DEDUPLICATION_KEYWORDS.items():
        if keyword in words_cpe_name:
            kw_count = words_cpe_name.count(keyword)
            del_idxs = []
            for i, word in enumerate(words_cpe_name):
                # allow keyword to appear max_count times max
                if kw_count < max_count + 1 or kw_count == len(words_cpe_name):
                    break
                if word == keyword:
                    del_idxs.append(i)
                    kw_count -= 1
            for i, idx in enumerate(del_idxs):
                del words_cpe_name[idx - i]

    # actually compute weights
    word_weights_cpe = {}
    for i, word in enumerate(words_cpe):
        if word not in word_weights_cpe:  # always use greatest weight
            word_weights_cpe[word] = math.exp(CPE_TERM_WEIGHT_EXP_FACTOR * i)

    word_weights_cpe_name = {}
    for i, word in enumerate(words_cpe_name):
        if word not in word_weights_cpe_name:  # always use greatest weight
            word_weights_cpe_name[word] = math.exp(CPE_TERM_WEIGHT_EXP_FACTOR * i)

    # compute CPE entry's cosine vector for similarity comparison
    cpe_tf = Counter(words_cpe + words_cpe_name)
    for term, tf in cpe_tf.items():
        cpe_tf[term] = tf / len(cpe_tf)
        if term in word_weights_cpe and term in word_weights_cpe_name:
            # average both obtained weights from CPE itself and its name
            cpe_tf[term] *= 0.5 * word_weights_cpe[term] + 0.5 * word_weights_cpe_name[term]
        elif term in word_weights_cpe:
            cpe_tf[term] *= word_weights_cpe[term]
        elif term in word_weights_cpe_name:
            cpe_tf[term] *= word_weights_cpe_name[term]

    cpe_abs = math.sqrt(sum([cnt**2 for cnt in cpe_tf.values()]))
    return cpe_tf, cpe_abs


def add_cpes_to_db(cpe_infos, config, check_duplicates=True):
    """
    Add new cpe_infos to DB (list of cpes or tuples of (cpe, name)).
    Assumes an existing CPE database.
    """

    # compute TF values
    cpe_infos_tf_norm = []
    for cpe_info in cpe_infos:
        cpe, cpe_name = "", ""
        if isinstance(cpe_info, list) or isinstance(cpe_info, tuple):
            cpe, cpe_name = cpe_info
        else:
            cpe = cpe_info
        cpe_tf, cpe_abs = compute_cpe_entry_tf_norm(cpe, cpe_name)
        cpe_infos_tf_norm.append((cpe, cpe_tf, cpe_abs))

    # get current max CPE entry ID
    db_conn = get_database_connection(config["DATABASE"])
    db_cursor = db_conn.cursor()
    db_cursor.execute("SELECT MAX(entry_id) FROM cpe_entries;")
    cur_max_eid = db_cursor.fetchone()[0]
    if not cur_max_eid:
        cur_max_eid = 0
    else:
        cur_max_eid += 1

    # add CPE infos to DB
    terms_to_entries = {}
    eid = cur_max_eid
    for cpe_info in cpe_infos_tf_norm:
        # insert unconditionally if fresh build, otherwise ensure entry doesn't exist yet
        do_insert = cur_max_eid == 0
        if check_duplicates and not do_insert:
            db_cursor.execute("SELECT * FROM cpe_entries where cpe = ?", (cpe_info[0],))
            do_insert = not bool(db_cursor.fetchone())

        if not check_duplicates or do_insert:
            db_cursor.execute(
                "INSERT INTO cpe_entries VALUES (?, ?, ?, ?)",
                (eid, cpe_info[0], ujson.dumps(cpe_info[1]), cpe_info[2]),
            )
            for term in cpe_info[1]:
                if term not in terms_to_entries:
                    terms_to_entries[term] = []
                terms_to_entries[term].append(eid)
            eid += 1

    db_conn.commit()
    db_cursor.close()
    db_cursor = db_conn.cursor()

    # add term --> entries translations to DB
    for term, entry_ids in terms_to_entries.items():
        if not entry_ids:
            continue

        # create new entry_ids_str
        i = 1
        entry_ids_str = str(entry_ids[0])
        while i < len(entry_ids):
            start_i = i
            while (i < len(entry_ids) - 1) and entry_ids[i] + 1 == entry_ids[i + 1]:
                i += 1
            if start_i == i:
                entry_ids_str += ",%d" % entry_ids[i]
            else:
                entry_ids_str += ",%d-%d" % (entry_ids[start_i], entry_ids[i])
            i += 1

        # check if term already exists in DB and if so add previous entry IDs
        db_cursor.execute("SELECT entry_ids FROM terms_to_entries where term = ?", (term,))
        prev_entry_ids = db_cursor.fetchone()
        do_insert = True
        if prev_entry_ids:
            prev_entry_ids = prev_entry_ids[0]
            if prev_entry_ids:
                entry_ids_str = prev_entry_ids + "," + entry_ids_str
                do_insert = False

        if entry_ids_str:
            if do_insert:
                db_cursor.execute(
                    "INSERT INTO terms_to_entries VALUES (?, ?)", (term, entry_ids_str)
                )
            else:
                db_cursor.execute(
                    "UPDATE terms_to_entries SET entry_ids = ? WHERE term = ?",
                    (entry_ids_str, term),
                )

    db_conn.commit()
    db_cursor.close()
    db_conn.close()


async def worker(headers, params, requestno, rate_limit, stop_update=[]):
    """Handles requests within its offset space asychronously, then performs processing steps to produce final database."""

    global UPDATE_SUCCESS

    async with rate_limit:
        api_data_response = await api_request(
            headers=headers, params=params, requestno=requestno
        )

    if not UPDATE_SUCCESS or stop_update:
        return None

    if api_data_response is not None:
        try:
            (cpes, deprecations) = intermediate_process(
                api_data=api_data_response, requestno=requestno
            )
            if DEBUG:
                print(f"[+] Performing calculations on request number {requestno}")

            cpe_infos = []
            for cpe in cpes:
                cpe_info = cpe.split(";")
                cpe_tf, cpe_abs = compute_cpe_entry_tf_norm(cpe_info[0], cpe_info[1])
                cpe_infos.append((cpe_info[0], cpe_tf, cpe_abs))
            return cpe_infos, deprecations
        except Exception as e:
            if UPDATE_SUCCESS and not SILENT:
                print(
                    "Got the following exception when downloading CPE data via API: %s" % str(e)
                )
            UPDATE_SUCCESS = False
            return None
    else:
        UPDATE_SUCCESS = False
        if not SILENT:
            print("api_data_response appears to be None.")
        return None


async def update(nvd_api_key=None, config=None, create_db=True, stop_update=[]):
    """Pulls current CPE data via the CPE API for an initial database build"""

    # import required modules in case they haven't been imported yet
    global asyncio, aiohttp, AsyncLimiter, requests
    import asyncio

    import aiohttp
    import requests
    from aiolimiter import AsyncLimiter

    if not SILENT:
        print("[+] Getting NVD's official CPE data (might take some time)")

    if not config:
        config = _load_config()

    if not nvd_api_key:
        nvd_api_key = os.getenv("NVD_API_KEY")
        if (not nvd_api_key) and config:
            nvd_api_key = config.get("NVD_API_KEY", None)

    if nvd_api_key:
        if not SILENT:
            print("[+] API Key found - Requests will be sent at a rate of 25 per 30s.")
        rate_limit = AsyncLimiter(25.0, 30.0)
        headers = {"apiKey": nvd_api_key}
    else:
        if not SILENT:
            print(
                "[-] No API Key found - Requests will be sent at a rate of 5 per 30s. To lower build time, consider getting an NVD API Key."
            )
        rate_limit = AsyncLimiter(5.0, 30.0)
        headers = {}

    # initial first request, also to set parameters
    offset = 0
    params = {"resultsPerPage": API_CPE_RESULTS_PER_PAGE, "startIndex": offset}
    numTotalResults = -1
    exception = ""
    for _ in range(3):
        try:
            cpe_api_data_page = requests.get(url=CPE_API_URL, headers=headers, params=params)
            numTotalResults = cpe_api_data_page.json().get("totalResults", -1)
            if numTotalResults > -1:
                break
        except Exception as e:  # e.g. json.decoder.JSONDecodeError
            exception = e
        time.sleep(1)

    if numTotalResults == -1:
        print(
            "Got the following exception when getting CPE count data via API: %s"
            % str(exception)
        )
        return False

    # make necessary amount of API requests to pull all CPE data
    requestno = 0
    tasks = []
    while offset <= numTotalResults:
        requestno += 1
        params = {"resultsPerPage": API_CPE_RESULTS_PER_PAGE, "startIndex": offset}
        task = asyncio.ensure_future(
            worker(
                headers=headers,
                params=params,
                requestno=requestno,
                rate_limit=rate_limit,
                stop_update=stop_update,
            )
        )
        tasks.append(task)
        offset += API_CPE_RESULTS_PER_PAGE

    while True:
        finished_tasks, pending_tasks = await asyncio.wait(
            tasks, return_when=asyncio.ALL_COMPLETED, timeout=2
        )
        if len(pending_tasks) < 1 or not UPDATE_SUCCESS or stop_update:
            break

    if not UPDATE_SUCCESS or stop_update:
        return False

    cpe_infos = []
    for task in finished_tasks:
        for cpe_triple in task.result()[0]:
            cpe_infos.append(cpe_triple)
    cpe_infos.sort(key=lambda cpe_info: cpe_info[0])

    db_type = config["DATABASE"]["TYPE"]
    db_name = config["DATABASE"]["NAME"]
    if not is_safe_db_name(db_name, db_type.lower()):
        print("Potentially malicious database name detected. Abort creation of database")
        return False

    if create_db:
        if db_type == "sqlite" and os.path.isfile(db_name):
            os.remove(db_name)
            os.makedirs(os.path.dirname(db_name), exist_ok=True)
        elif db_type == "mariadb":
            db_conn = get_database_connection(config["DATABASE"], db_name="")
            db_cursor = db_conn.cursor()
            db_cursor.execute(f"CREATE OR REPLACE DATABASE {db_name};")
            db_cursor.execute(f"use {db_name};")
            db_conn.commit()
            db_cursor.close()
            db_conn.close()

    db_conn = get_database_connection(config["DATABASE"])
    db_cursor = db_conn.cursor()

    # create tables
    with open(CREATE_SQL_STATEMENTS_FILE) as f:
        create_sql_statements = ujson.loads(f.read())
    db_cursor.execute(create_sql_statements["TABLES"]["CPE_ENTRIES"][db_type])
    db_cursor.execute(create_sql_statements["TABLES"]["TERMS_TO_ENTRIES"][db_type])
    db_conn.commit()
    db_cursor.close()
    db_cursor = db_conn.cursor()

    # add CPE infos to DB
    terms_to_entries = {}
    products_cpe_count = {}
    for i, cpe_info in enumerate(cpe_infos):
        product_cpe = ":".join(cpe_info[0].split(":")[:5]) + ":"
        if product_cpe not in products_cpe_count:
            products_cpe_count[product_cpe] = 0
        products_cpe_count[product_cpe] += 1

        db_cursor.execute(
            "INSERT INTO cpe_entries VALUES (?, ?, ?, ?)",
            (i, cpe_info[0], ujson.dumps(cpe_info[1]), cpe_info[2]),
        )
        for term in cpe_info[1]:
            if term not in terms_to_entries:
                terms_to_entries[term] = []
            terms_to_entries[term].append(i)
    db_conn.commit()
    db_cursor.close()
    db_cursor = db_conn.cursor()

    # insert products_cpe_count into DB
    if config["DATABASE"]["TYPE"] == "sqlite":
        db_cursor.execute("DROP TABLE IF EXISTS product_cpe_counts;")
        create_counts_table = "CREATE TABLE product_cpe_counts (product_cpe_prefix VARCHAR(255), count INTEGER, PRIMARY KEY (product_cpe_prefix));"
    elif config["DATABASE"]["TYPE"] == "mariadb":
        create_counts_table = "CREATE OR REPLACE TABLE product_cpe_counts (product_cpe_prefix VARCHAR(255) CHARACTER SET ascii, count INTEGER, PRIMARY KEY (product_cpe_prefix));"
    db_cursor.execute(create_counts_table)

    for product_cpe, count in products_cpe_count.items():
        db_cursor.execute("INSERT INTO product_cpe_counts VALUES(?, ?)", (product_cpe, count))

    # add term --> entries translations to DB
    for term, entry_ids in terms_to_entries.items():
        if not entry_ids:
            continue

        i = 1
        entry_ids_str = str(entry_ids[0])
        while i < len(entry_ids):
            start_i = i
            while (i < len(entry_ids) - 1) and entry_ids[i] + 1 == entry_ids[i + 1]:
                i += 1
            if start_i == i:
                entry_ids_str += ",%d" % entry_ids[i]
            else:
                entry_ids_str += ",%d-%d" % (entry_ids[start_i], entry_ids[i])
            i += 1
        db_cursor.execute("INSERT INTO terms_to_entries VALUES (?, ?)", (term, entry_ids_str))

    db_conn.commit()
    db_cursor.close()
    db_conn.close()

    if stop_update:
        return False

    # create CPE deprecations file
    os.makedirs(os.path.dirname(config["DEPRECATED_CPES_FILE"]), exist_ok=True)
    with open(config["DEPRECATED_CPES_FILE"], "w") as outfile:
        final_deprecations = {}
        for task in finished_tasks:
            for deprecation in task.result()[1]:
                deprecated_cpe = list(deprecation)[0]
                if deprecated_cpe in final_deprecations:
                    final_deprecations[deprecated_cpe] = list(
                        set(final_deprecations[deprecated_cpe] + deprecation[deprecated_cpe])
                    )
                else:
                    final_deprecations[deprecated_cpe] = deprecation[deprecated_cpe]
        outfile.write("%s\n" % ujson.dumps(final_deprecations))
        outfile.close()

    return True


def get_possible_versions_in_query(query):
    version_parts = []
    version_str_match = VERSION_MATCH_CPE_CREATION_RE.search(query)
    if version_str_match:
        full_version_str = version_str_match.group(1).strip()
        version_parts.append(full_version_str)
        version_parts += re.split(r"[\+\-\_\~ ]", full_version_str)

        # remove first element in case of duplicate
        if len(version_parts) > 1 and version_parts[0] == version_parts[1]:
            version_parts = version_parts[1:]
    return version_parts


def _get_alternative_queries(init_queries):
    alt_queries_mapping = {}
    for query in init_queries:
        alt_queries_mapping[query] = []

        # replace 'httpd' with 'http' e.g. for Apache HTTP Server
        if "httpd" in query:
            alt_query = query.replace("httpd", "http")
            alt_queries_mapping[query].append(alt_query)

        # check for "simple" abbreviations
        alt_query_all_replaced = query
        for abbreviation, (required_keywords, replacement) in QUERY_ABBREVIATIONS.items():
            if not required_keywords or any(keyword in query for keyword in required_keywords):
                if (
                    query.startswith(abbreviation)
                    or query.endswith(abbreviation)
                    or f" {abbreviation} " in query
                ):
                    alt_queries_mapping[query].append(
                        query.replace(abbreviation, f" {replacement} ")
                    )
                    alt_query_all_replaced = alt_query_all_replaced.replace(
                        abbreviation, f" {replacement} "
                    )

        if alt_query_all_replaced != query:
            alt_queries_mapping[query].append(alt_query_all_replaced)

        # check for Cisco 'CM' and 'SME' abbreviations
        if "cisco" in query and (
            query.startswith("cm ") or query.endswith(" cm") or " cm " in query
        ):
            alt_query = query.replace("cm", "communications manager")
            if "sm" in query:
                alt_query = alt_query.replace("sm", "session management")
            alt_queries_mapping[query].append(alt_query)

        # fix popular queries, where the search algorithm has difficulties with sub products
        # e.g. is the query 'flask' looking for THE 'Flask' by the PalletsProjects or the
        # 'Flask' plugin 'Flask Caching' by 'Flask Caching Project'?
        for product, helper_query in POPULAR_QUERY_CORRECTIONS.items():
            if product in query and not any(word in query for word in helper_query.split(" ")):
                alt_queries_mapping[query].append(helper_query + " " + query)

        # check for different variants of js library names, e.g. 'moment.js' vs. 'momentjs' vs. 'moment js'
        query_words = query.split()
        if "js " in query or " js" in query or query.endswith("js"):
            alt_queries = []
            for i, word in enumerate(query_words):
                word = word.strip()
                new_query_words1, new_query_words2 = [], []
                if word == "js" and i > 0:
                    new_query_words1 = query_words[: i - 1] + [query_words[i - 1] + "js"]
                    new_query_words2 = query_words[: i - 1] + [query_words[i - 1] + ".js"]
                elif word.endswith(".js") or word.endswith("js"):
                    if i > 0:
                        new_query_words1 += query_words[:i]
                        new_query_words2 += query_words[:i]
                    if word.endswith(".js"):
                        new_query_words1 += [word[: -len(".js")]] + ["js"]
                        new_query_words2 += [word[: -len(".js")] + "js"]
                    else:
                        new_query_words1 += [word[: -len("js")]] + ["js"]
                        new_query_words2 += [word[: -len("js")] + ".js"]

                if new_query_words1:
                    if i < len(query_words) - 1:
                        new_query_words1 += query_words[i + 1 :]
                        new_query_words2 += query_words[i + 1 :]
                    alt_queries.append(" ".join(new_query_words1))
                    alt_queries.append(" ".join(new_query_words2))

            if alt_queries:
                alt_queries_mapping[query] += alt_queries

        # check for a version containing a commit-ID, date, etc.
        version_parts = get_possible_versions_in_query(query)
        if len(version_parts) > 1:  # first item is always the entire version string
            query_no_version = query.replace(version_parts[0], "")
            alt_queries_mapping[query].append(query_no_version + " ".join(version_parts[1:]))

        # split version parts with different character groups by a space,
        # e.g. 'openssh 7.4p1' --> 'openssh 7.4 p1'
        pot_alt_query = ""
        cur_char_class = string.ascii_letters
        did_split, seen_first_break = False, False
        splits, maxsplit = 0, query.count(" ") + ALT_QUERY_MAXSPLIT
        for char in query:
            if char in (" ", ".", "-", "+"):
                seen_first_break = True
                pot_alt_query += char
                did_split = False
                continue

            if (
                seen_first_break
                and splits < maxsplit
                and char not in cur_char_class
                and not did_split
            ):
                pot_alt_query += " "
                did_split = True
                splits += 1
                if char in string.ascii_letters:
                    cur_char_class = string.ascii_letters
                else:
                    try:
                        int(char)
                        cur_char_class = "0123456789"
                    except ValueError:
                        cur_char_class = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            pot_alt_query += char

        pot_alt_query_parts = pot_alt_query.split()
        for i in range(len(pot_alt_query_parts)):
            if pot_alt_query_parts[i][-1] in (".", "-", "+"):
                pot_alt_query_parts[i] = pot_alt_query_parts[i][:-1]
        pot_alt_query = " ".join(pot_alt_query_parts)

        if pot_alt_query != query.strip():
            alt_queries_mapping[query].append(pot_alt_query)  # w/o including orig query words
            for word in query.split():
                if word not in pot_alt_query:
                    pot_alt_query += " " + word
            alt_queries_mapping[query].append(pot_alt_query)  # w/ including orig query words

        # add alt query in case likely subversion is split from main version by a space
        if len(query_words) > 2 and len(query_words[-1]) < 7:
            alt_queries_mapping[query].append(query + " " + query_words[-2] + query_words[-1])
            alt_queries_mapping[query].append(
                " ".join(query_words[:-2]) + " " + query_words[-2] + query_words[-1]
            )

        # zero extend versions, e.g. 'Apache httpd 2.4' --> 'Apache httpd 2.4.0'
        version_match = VERSION_MATCH_ZE_RE.search(query)
        if version_match:
            alt_queries_mapping[query].append(
                query.replace(version_match.group(0), version_match.group(0) + ".0")
            )
            alt_queries_mapping[query].append(
                query.replace(version_match.group(0), version_match.group(0) + ".0.0")
            )

    return alt_queries_mapping


def _search_cpes(queries_raw, db_cursor=None, count=None, threshold=None, config=None):
    """Facilitate CPE search as specified by the program arguments"""

    if not config:
        config = _load_config()

    if count is None:
        count = int(config["CPE_SEARCH_COUNT"])
    if threshold is None:
        threshold = float(config["CPE_SEARCH_THRESHOLD"])

    # create term frequencies and normalization factors for all queries
    queries = [query.lower() for query in queries_raw]

    # add alternative queries to improve retrieval
    alt_queries_mapping = _get_alternative_queries(queries)
    for alt_queries in alt_queries_mapping.values():
        queries += alt_queries

    query_infos = {}
    most_similar = {}
    all_query_words = set()
    included_word_sets = {}
    for query in queries:
        words_query = TEXT_TO_VECTOR_RE.findall(query)
        if words_query in included_word_sets.values():
            continue
        included_word_sets[query] = words_query
        word_weights_query = {}
        for i, word in enumerate(words_query):
            if word not in word_weights_query:
                word_weights_query[word] = math.exp(QUERY_TERM_WEIGHT_EXP_FACTOR * i)

        # compute query's cosine vector for similarity comparison
        query_tf = Counter(words_query)
        for term, tf in query_tf.items():
            query_tf[term] = word_weights_query[term] * (tf / len(query_tf))
        all_query_words |= set(query_tf.keys())
        query_abs = math.sqrt(sum([cnt**2 for cnt in query_tf.values()]))
        query_infos[query] = (query_tf, query_abs)
        most_similar[query] = {}
    queries = list(included_word_sets.keys())

    # set up DB connector
    close_db_cursor = False
    if not db_cursor:
        conn = get_database_connection(config["DATABASE"])
        db_cursor = conn.cursor()
        close_db_cursor = True

    # figure out which CPE infos are relevant, based on the terms of all queries
    all_cpe_entry_ids = []
    for word in all_query_words:
        # query can only return one result, b/c term is PK
        db_query = "SELECT entry_ids FROM terms_to_entries WHERE term = ?"
        db_cursor.execute(db_query, (word,))
        if not db_cursor:
            continue

        cpe_entry_ids = db_cursor.fetchall()
        if not cpe_entry_ids:
            continue
        cpe_entry_ids = cpe_entry_ids[0][0].split(",")
        all_cpe_entry_ids.append(int(cpe_entry_ids[0]))

        for eid in cpe_entry_ids[1:]:
            if "-" in eid:
                eid = eid.split("-")
                all_cpe_entry_ids += list(range(int(eid[0]), int(eid[1]) + 1))
            else:
                all_cpe_entry_ids.append(int(eid))

    # iterate over all retrieved CPE infos and find best matching CPEs for queries
    all_cpe_infos = []
    # limiting number of max_results_per_query boosts performance of MariaDB
    max_results_per_query = 1000
    remaining = len(all_cpe_entry_ids)

    while remaining > 0:
        count_params_in_str = min(remaining, max_results_per_query)
        param_in_str = ("?," * count_params_in_str)[:-1]

        db_query = (
            "SELECT cpe, term_frequencies, abs_term_frequency FROM cpe_entries WHERE entry_id IN (%s)"
            % param_in_str
        )
        db_cursor.execute(
            db_query, all_cpe_entry_ids[remaining - count_params_in_str : remaining]
        )
        cpe_infos = []
        if db_cursor:
            cpe_infos = db_cursor.fetchall()
        all_cpe_infos += cpe_infos
        remaining -= max_results_per_query

    # same order needed for test repeatability
    if os.environ.get("IS_CPE_SEARCH_TEST", "false") == "true":
        all_cpe_infos = sorted(all_cpe_infos)

    # Search Idea: Divide and Conquer
    # Divide all CPEs into a dict of CPE classes where only the most similar
    # CPE for every class is stored. In the end, unify all of these most similar
    # entries and sort them by similarity
    # (cpe-class: base CPE + number of non-wildcard fields)
    processed_cpes = set()
    for cpe_info in all_cpe_infos:
        cpe, cpe_tf, cpe_abs = cpe_info

        # all_cpe_infos may contain duplicates
        if cpe in processed_cpes:
            continue
        processed_cpes.add(cpe)

        cpe_tf = ujson.loads(cpe_tf)
        cpe_abs = float(cpe_abs)

        for query in queries:
            query_tf, query_abs = query_infos[query]
            intersecting_words = set(cpe_tf.keys()) & set(query_tf.keys())
            inner_product = sum([cpe_tf[w] * query_tf[w] for w in intersecting_words])

            normalization_factor = cpe_abs * query_abs

            if not normalization_factor:  # avoid divison by 0
                continue

            sim_score = float(inner_product) / float(normalization_factor)

            if threshold > 0 and sim_score < threshold:
                continue

            cpe_base = ":".join(cpe.split(":")[:5]) + ":"
            cpe_class = (
                cpe_base
                + "-"
                + str(10 - sum(cpe_field in ("*", "-", "") for cpe_field in cpe.split(":")))
            )
            if (
                cpe_class not in most_similar[query]
                or sim_score > most_similar[query][cpe_class][1]
            ):
                most_similar[query][cpe_class] = (cpe, sim_score)

    # unify the individual most similar results
    for query in queries:
        unified_most_similar = set()
        if most_similar[query]:
            for cpe, sim_score in most_similar[query].values():
                unified_most_similar.add((cpe, sim_score))
            most_similar[query] = sorted(
                unified_most_similar, key=lambda entry: (-entry[1], entry[0])
            )

    # only return the number of requested CPEs (per query including alt queries)
    if count != -1:
        for query in queries:
            if most_similar[query]:
                most_similar[query] = most_similar[query][:count]
            else:
                most_similar[query] = []

    # create intermediate results (including any additional queries)
    intermediate_results = {}
    for query in queries:
        if not most_similar[query] or (
            len(most_similar[query]) == 1 and most_similar[query][0][1] == -1
        ):
            continue

        intermediate_results[query] = most_similar[query]

        rm_idxs = []
        for i, result in enumerate(intermediate_results[query]):
            if result[1] == -1:
                rm_idxs.append(i)

        for i in rm_idxs:
            del intermediate_results[query][i]

    # create final results
    results = {}
    for query_raw in queries_raw:
        query = query_raw.lower()

        if query not in intermediate_results and (
            query not in alt_queries_mapping or not alt_queries_mapping[query]
        ):
            continue

        if query not in alt_queries_mapping or not alt_queries_mapping[query]:
            results[query_raw] = intermediate_results[query]
        else:
            unified_most_similar = set()
            if most_similar[query]:
                unified_most_similar = set(intermediate_results[query])
                for alt_query in alt_queries_mapping[query]:
                    if alt_query != query:
                        unified_most_similar |= set(intermediate_results.get(alt_query, []))

            results[query_raw] = sorted(
                unified_most_similar, key=lambda entry: (-entry[1], entry[0])
            )

    # remove duplicates coming from alt queries and keep highest score
    for query_raw in queries_raw:
        retrieved_cpes = set()
        del_idxs = []
        if query_raw not in results:
            continue
        for i in range(len(results[query_raw])):
            if results[query_raw][i][0] in retrieved_cpes:
                del_idxs.append(i)
            else:
                retrieved_cpes.add(results[query_raw][i][0])
        for i in range(len(del_idxs)):
            del results[query_raw][del_idxs[i] - i]

    # only return the number of requested CPEs for final results
    if count != -1:
        for query in queries_raw:
            if results.get(query):
                results[query] = results[query][:count]
            else:
                results[query] = []
            results[query]

    # close cursor and connection afterwards
    if close_db_cursor:
        db_cursor.close()
        conn.close()

    return results


def is_cpe_equal(cpe1, cpe2):
    """Return True if both CPEs are considered equal, False otherwise"""

    if len(cpe1) != len(cpe2):
        return False

    for i in range(len(cpe1)):
        if cpe1[i] != cpe2[i]:
            if not (cpe1[i] in ("*", "-") and cpe2[i] in ("*", "-")):
                return False
    return True


def match_cpe23_to_cpe23_from_dict(cpe23_in, config=None, db_cursor=None):
    """
    Try to return a valid CPE 2.3 string that exists in the NVD's CPE
    dictionary based on the given, potentially badly formed, CPE string.
    """

    if not config:
        config = _load_config()

    # if the given CPE is simply not a full CPE 2.3 string
    pot_new_cpe = ""
    if cpe23_in.count(":") < 12:
        pot_new_cpe = cpe23_in
        if pot_new_cpe.endswith(":"):
            pot_new_cpe += "*"
        while pot_new_cpe.count(":") < 12:
            pot_new_cpe += ":*"

    pre_cpe_in = cpe23_in
    all_cpes = get_all_cpes(config, db_cursor)
    while pre_cpe_in.count(":") > 3:  # break if next cpe part would be vendor part
        pre_cpe_in = pre_cpe_in[:-1]
        if (
            pre_cpe_in.endswith(":") or pre_cpe_in.count(":") > 9
        ):  # skip rear parts in fixing process
            continue

        for cpe in all_cpes:
            if cpe23_in == cpe:
                return cpe23_in
            if pot_new_cpe and pot_new_cpe == cpe:
                return pot_new_cpe

            if pre_cpe_in in cpe:
                # stitch together the found prefix and the remaining part of the original CPE
                if cpe23_in[len(pre_cpe_in)] == ":":
                    cpe_in_add_back = cpe23_in[len(pre_cpe_in) + 1 :]
                else:
                    cpe_in_add_back = cpe23_in[len(pre_cpe_in) :]
                new_cpe = "%s:%s" % (pre_cpe_in, cpe_in_add_back)

                # get new_cpe to full CPE 2.3 length by adding or removing wildcards
                while new_cpe.count(":") < 12:
                    new_cpe += ":*"
                if new_cpe.count(":") > 12:
                    new_cpe = new_cpe[: new_cpe.rfind(":")]

                # if a matching CPE was found, return it
                if is_cpe_equal(new_cpe, cpe):
                    return cpe
    return ""


def create_cpes_from_base_cpe_and_query(cpe, query):
    new_cpes = []
    version_parts = get_possible_versions_in_query(query)

    # create CPEs where version parts are put into subsequent CPE fields
    if len(version_parts) > 2:
        for i in range(1, len(version_parts)):
            cpe_parts = cpe.split(":")
            cpe_parts = cpe_parts[:5] + version_parts[1 : i + 1] + cpe_parts[5 + i :]
            new_cpes.append(":".join(cpe_parts))

    # check if there is only one complex version without a distinct seperator
    # and if so put the two parts into the proper CPE fields (e.g. 10.4p18 --> 10.4 p18)
    if len(version_parts) == 1:
        complex_version_match = VERSION_SPLIT_DIFF_CHARSETS_RE.search(version_parts[0])
        if complex_version_match:
            split_idx = complex_version_match.start()
            ver_part1 = version_parts[0][:split_idx]
            ver_part2 = version_parts[0][split_idx:]
            while ver_part2 and not ver_part2[0].isalnum():
                ver_part2 = ver_part2[1:]
            if ver_part2:
                cpe_parts = cpe.split(":")
                cpe_parts[5] = ver_part1
                cpe_parts[6] = ver_part2
                new_cpes.append(":".join(cpe_parts))

    # check whether a subversion part (starting at seventh CPE field) is already in CPE ...
    version_part_in_cpe = False
    for i, version in enumerate(version_parts[2:]):
        cpe_parts = cpe.split(":")
        if version in cpe_parts[6 + i :]:
            version_part_in_cpe = True
            break

    # check that no cpe subversion part is already in query version part
    cpe_part_in_version = False
    for cpe_part in cpe.split(":")[6:]:
        if version_parts and cpe_part in version_parts[0]:
            cpe_part_in_version = True
            break

    # ... and if not, create CPE where the detected version string is entirely put into the sixth CPE field
    if version_parts and not version_part_in_cpe and not cpe_part_in_version:
        cpe_parts = cpe.split(":")
        cpe_parts[5] = version_parts[0].replace(" ", "_")

        # ... also remove any more specific CPE parts not in the query
        for i in range(6, len(cpe_parts)):
            found_part = False
            for sep in (' ', '-', '+', '_'):
                if cpe_parts[i] + sep in query or sep + cpe_parts[i] in query:
                    found_part = True
                    break
            if not found_part:
                cpe_parts[i] = '*'

        new_cpes.append(":".join(cpe_parts))

    return new_cpes


def is_versionless_query(query):
    version_str_match = VERSION_MATCH_CPE_CREATION_RE.search(query)
    if not version_str_match:
        return True
    return False


def create_base_cpe_if_versionless_query(cpe, query):
    if is_versionless_query(query):
        cpe_parts = cpe.split(":")
        base_cpe = ":".join(cpe_parts[:5] + ["*"] * 8)
        return base_cpe

    return None


def get_all_cpes(config=None, db_cursor=None):

    if not config:
        config = _load_config()

    close_db_cursor = False
    if not close_db_cursor:
        conn = get_database_connection(config["DATABASE"])
        db_cursor = conn.cursor()
        close_db_cursor = True

    db_cursor.execute("SELECT cpe FROM cpe_entries")
    cpes = [cpe[0] for cpe in db_cursor]

    if close_db_cursor:
        db_cursor.close()
        conn.close()

    return cpes


def cpe_matches_query(cpe, query):
    """Return True if the given CPE somewhat matches the query, e.g. its version number."""

    check_str, bad_match = cpe[8:], False

    # ensure that the retrieved CPE has a number if query has a number
    if any(char.isdigit() for char in query) and not any(char.isdigit() for char in check_str):
        bad_match = True

    # if a version number is clearly detectable in query, ensure this version is somewhat reflected in the CPE
    versions_in_query = get_possible_versions_in_query(query)
    if not bad_match:
        cpe_has_matching_version = False
        for possible_version in versions_in_query:
            # ensure version has at least two parts to avoid using a short version for checking
            if "." not in possible_version:
                continue

            idx_pos_ver, idx_check_str = 0, 0
            while idx_pos_ver < len(possible_version) and idx_check_str < len(check_str):
                while (
                    idx_pos_ver < len(possible_version)
                    and not possible_version[idx_pos_ver].isdigit()
                ):
                    idx_pos_ver += 1
                if (
                    idx_pos_ver < len(possible_version)
                    and possible_version[idx_pos_ver] == check_str[idx_check_str]
                ):
                    idx_pos_ver += 1
                idx_check_str += 1

            if idx_pos_ver == len(possible_version):
                cpe_has_matching_version = True
                break

        if versions_in_query and not cpe_has_matching_version:
            bad_match = True

    # check that at least one query term, apart from the version number, is contained in the CPE
    if not bad_match:
        non_version_terms = [
            term.lower() for term in SPLIT_QUERY_TERMS_RE.split(query) if term not in versions_in_query
        ]
        if not any(term in cpe for term in non_version_terms):
            bad_match = True

    return not bad_match


def search_cpes(query, db_cursor=None, count=None, threshold=None, config=None):
    if not query:
        return {"cpes": [], "pot_cpes": []}

    if not config:
        config = _load_config()

    if count is None:
        count = int(config["CPE_SEARCH_COUNT"])
    if threshold is None:
        threshold = float(config["CPE_SEARCH_THRESHOLD"])

    query = query.strip()
    cpes, pot_cpes = [], []

    if not MATCH_CPE_23_RE.match(query):
        cpes = _search_cpes(
            [query], db_cursor, count=count, threshold=CPE_SEARCH_THRESHOLD_ALT, config=config
        )
        cpes = cpes.get(query, [])

        if not cpes:
            return {"cpes": [], "pot_cpes": []}

        # try to clean stop words / symbols
        cpe_creation_query = CPE_CREATION_DEL_SYMBOLS_RE.sub(" ", query)
        cpe_creation_query = cpe_creation_query.replace("  ", " ")

        # always create related queries with supplied version number
        for cpe, sim in cpes:
            new_cpes = create_cpes_from_base_cpe_and_query(cpe, cpe_creation_query)
            for new_cpe in new_cpes:
                # do not overwrite sim score of an existing CPE
                if any(is_cpe_equal(new_cpe, existing_cpe[0]) for existing_cpe in cpes):
                    continue

                # only add CPE if it was not seen before
                if new_cpe and (
                    not pot_cpes
                    or not any(is_cpe_equal(new_cpe, other[0]) for other in pot_cpes)
                ):
                    pot_cpes.append((new_cpe, -1 * sim))

            if not any(is_cpe_equal(cpe, other[0]) for other in pot_cpes):
                pot_cpes.append((cpe, sim))

        # always create related queries without version number if query is versionless
        versionless_cpe_inserts, new_idx = [], 0
        for cpe, _ in pot_cpes:
            base_cpe = create_base_cpe_if_versionless_query(cpe, cpe_creation_query)
            if base_cpe:
                if (
                    not any(is_cpe_equal(base_cpe, other[0]) for other in pot_cpes)
                ) and not any(
                    is_cpe_equal(base_cpe, other[0][0]) for other in versionless_cpe_inserts
                ):
                    versionless_cpe_inserts.append(((base_cpe, -1 * sim), new_idx))
                    new_idx += 1
            new_idx += 1

        for new_cpe, idx in versionless_cpe_inserts:
            pot_cpes.insert(idx, new_cpe)

        # catch and filter out bad CPE matches
        prev_cpe_count = len(cpes)
        cpes = [cpe for cpe in cpes if cpe_matches_query(cpe[0], cpe_creation_query)]

        # break early on bad match
        if prev_cpe_count != len(cpes):
            if cpes and cpes[0][1] > threshold:
                return {"cpes": cpes, "pot_cpes": pot_cpes}
            else:
                # filter out completely irrelevant CPEs
                new_pot_cpes = []
                for pot_cpe in pot_cpes:
                    if any(
                        word.lower() in query.lower() for word in pot_cpe[0].split(":")[3:5]
                    ):
                        new_pot_cpes.append(pot_cpe)
                return {"cpes": [], "pot_cpes": new_pot_cpes}

        # also catch bad match if query is versionless, but retrieved CPE is not
        cpe_version = cpes[0][0].split(":")[5] if cpes[0][0].count(":") > 5 else ""
        if cpe_version not in ("*", "-"):
            base_cpe = create_base_cpe_if_versionless_query(cpes[0][0], query)
            if base_cpe:
                # remove CPEs from related queries that have a version
                pot_cpes_versionless = []
                for _, (pot_cpe, score) in enumerate(pot_cpes):
                    cpe_version_iter = pot_cpe.split(":")[5] if pot_cpe.count(":") > 5 else ""
                    if cpe_version_iter in ("", "*", "-"):
                        pot_cpes_versionless.append((pot_cpe, score))

                return {"cpes": [], "pot_cpes": pot_cpes_versionless}

        if cpes[0][1] < threshold:
            cpes = []
    else:
        pot_cpes = []

    return {"cpes": cpes, "pot_cpes": pot_cpes}


def main():
    global SILENT, UPDATE_SUCCESS

    SILENT = not sys.stdout.isatty()
    args = parse_args()

    if args.verbose:
        SILENT = False

    perform_update = False
    if args.update:
        perform_update = True

    config = _load_config(args.config)
    if (
        args.queries
        and config["DATABASE"]["TYPE"] == "sqlite"
        and not os.path.isfile(config["DATABASE"]["NAME"])
    ):
        if not SILENT:
            print("[+] Running initial setup (might take a couple of minutes)", file=sys.stderr)
        perform_update = True

    if perform_update:
        import asyncio

        import aiohttp
        import requests
        from aiolimiter import AsyncLimiter

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update(args.api_key, config))

        if not UPDATE_SUCCESS:
            print("[-] Failed updating the local CPE database!")

    if args.queries:
        results = {}
        for query in args.queries:
            results[query] = search_cpes(query, None, args.number, -1, config).get("cpes", [])

        if not results:
            print()
            print(results)

        for i, query in enumerate(results):
            if not SILENT and i > 0:
                print()

            if results[query]:
                print(results[query][0][0])
                if not SILENT:
                    pprint.pprint(results[query])
            else:
                print("Could not find software for query: %s" % query)
                if not SILENT:
                    pprint.pprint([])


if __name__ == "__main__":
    main()
