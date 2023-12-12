#!/usr/bin/env python3

import argparse
from collections import Counter
import math
import os
import pprint
import re
import sqlite3
import string
import sys
import time


try:  # use ujson if available
    import ujson as json
except ModuleNotFoundError:
    import json


# Constants
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CPE_API_URL = "https://services.nvd.nist.gov/rest/json/cpes/2.0/"
CPE_DATABASE_FILE = os.path.join(SCRIPT_DIR, "cpe-search-dictionary.db3")
DEPRECATED_CPES_FILE = os.path.join(SCRIPT_DIR, "deprecated-cpes.json")
DB_URI, DB_CONN_MEM = 'file:cpedb?mode=memory&cache=shared', None
TEXT_TO_VECTOR_RE = re.compile(r"[\w+\.]+")
GET_ALL_CPES_RE = re.compile(r'(.*);.*;.*')
VERSION_MATCH_ZE_RE = re.compile(r'\b([\d]+\.?){1,4}\b')
VERSION_MATCH_CPE_CREATION_RE = re.compile(r'\b((\d+[\.\-]?){1,4}([a-z\d]{0,3})?)[^\w]*$')
TERMS = []
TERMS_MAP = {}
ALT_QUERY_MAXSPLIT = 1
API_CPE_RESULTS_PER_PAGE = 10000
UPDATE_SUCCESS = True
SILENT = True
DEBUG = False


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)")
    parser.add_argument("-u", "--update", action="store_true", help="Update the local CPE database")
    parser.add_argument("-k", "--api-key", type=str, help="NVD API key to use for updating the local CPE dictionary")
    parser.add_argument("-c", "--count", default=3, type=int, help="The number of CPEs to show in the similarity overview (default: 3)")
    parser.add_argument("-q", "--query", dest="queries", metavar="QUERY", action="append", help="A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose and print status information")

    args = parser.parse_args()
    if not args.update and not args.queries:
        parser.print_help()
    return args


async def api_request(headers, params, requestno):
    '''Perform request to API for one task'''

    global UPDATE_SUCCESS

    if not UPDATE_SUCCESS:
        return None

    retry_limit = 3
    retry_interval = 6
    for _ in range(retry_limit + 1):
        async with aiohttp.ClientSession() as session:
            try:
                cpe_api_data_response = await session.get(url=CPE_API_URL, headers=headers, params=params)
                if cpe_api_data_response.status == 200:
                    if DEBUG:
                        print(f"[+] Successfully received data from request {requestno}.")
                    return await cpe_api_data_response.json()
                else:
                    if DEBUG:
                        print(f"[-] Received status code {cpe_api_data_response.status} on request {requestno} Retrying...")
                await asyncio.sleep(retry_interval)
            except Exception as e:
                if UPDATE_SUCCESS and not SILENT:
                    print('Got the following exception when downloading CPE data via API: %s' % str(e))
    UPDATE_SUCCESS = False
    return None


def intermediate_process(api_data, requestno):
    '''Performs extraction of CPE names and CPE titles'''

    if DEBUG:
        print(f"[+] Intermediate processing on request number {requestno}")

    products = api_data.get('products')
    cpes = []
    deprecations = []
    for product in products:
        extracted_title = ""
        deprecated = product['cpe']['deprecated']
        cpe_name = product['cpe']['cpeName']
        for title in product['cpe']['titles']:
            # assume an english title is always present
            if title['lang'] == 'en':
                extracted_title = title['title']
        if deprecated:
            deprecated_by = {cpe_name:[]}
            for item in product['cpe']['deprecatedBy']:
                deprecated_by[cpe_name].append(item.get('cpeName'))
            deprecations.append(deprecated_by)
        cpes.append(str(cpe_name + ';' + extracted_title + ';'))
    return cpes, deprecations


def perform_calculations(cpes, requestno):
    '''Performs calculations for searching for CPEs on every CPE in a given CPE list'''

    if DEBUG:
        print(f"[+] Performing calculations on request number {requestno}")

    cpe_info = []
    for cpe in cpes:
        cpe_mod = cpe.split(';')[0].replace("_", ":").replace("*", "").replace("\\", "")
        cpe_name = cpe.split(';')[1].lower()
        cpe_name_elems = [word for word in cpe_name.split()]
        cpe_elems = [cpe_part for cpe_part in cpe_mod[10:].split(':') if cpe_part != ""]
        words = TEXT_TO_VECTOR_RE.findall(" ".join(cpe_elems + cpe_name_elems))
        cpe_tf = Counter(words)
        for term, tf in cpe_tf.items():
            cpe_tf[term] = tf / len(cpe_tf)
        cpe_abs = math.sqrt(sum([cnt**2 for cnt in cpe_tf.values()]))
        cpe_info.append((cpe.split(';')[0].lower(), cpe_tf, cpe_abs))
    return cpe_info


async def worker(headers, params, requestno, rate_limit):
    '''Handles requests within its offset space asychronously, then performs processing steps to produce final database.'''

    global UPDATE_SUCCESS

    async with rate_limit:
        api_data_response = await api_request(headers=headers, params=params, requestno=requestno)

    if not UPDATE_SUCCESS:
        return None

    if api_data_response is not None:
        try:
            (cpes, deprecations) = intermediate_process(api_data=api_data_response, requestno=requestno)
            cpe_infos = perform_calculations(cpes=cpes, requestno=requestno), deprecations
            return cpe_infos
        except Exception as e:
            if UPDATE_SUCCESS and not SILENT:
                print('Got the following exception when downloading CPE data via API: %s' % str(e))
            UPDATE_SUCCESS = False
            return None
    else:
        UPDATE_SUCCESS = False
        if not SILENT:
            print('api_data_response appears to be None.')
        return None


async def update(nvd_api_key=None):
    '''Pulls current CPE data via the CPE API for an initial database build'''

    # import required modules in case they haven't been imported yet
    global asyncio, aiohttp, AsyncLimiter, requests
    import asyncio
    import aiohttp
    from aiolimiter import AsyncLimiter
    import requests

    if not SILENT:
        print("[+] Getting NVD's official CPE data (might take some time)")

    if not nvd_api_key:
        nvd_api_key = os.getenv('NVD_API_KEY')

    if nvd_api_key:
        if not SILENT:
            print('[+] API Key found - Requests will be sent at a rate of 25 per 30s.')
        rate_limit = AsyncLimiter(25.0, 30.0)
        headers = {'apiKey': nvd_api_key}
    else:
        if not SILENT:
            print('[-] No API Key found - Requests will be sent at a rate of 5 per 30s. To lower build time, consider getting an NVD API Key.')
        rate_limit = AsyncLimiter(5.0, 30.0)
        headers = {}

    # initial first request, also to set parameters
    offset = 0
    params = {'resultsPerPage': API_CPE_RESULTS_PER_PAGE, 'startIndex': offset}
    numTotalResults = -1
    exception = ''
    for _ in range(3):
        try:
            cpe_api_data_page = requests.get(url=CPE_API_URL, headers=headers, params=params)
            numTotalResults = cpe_api_data_page.json().get('totalResults', -1)
            if numTotalResults > -1:
                break
        except Exception as e:  # e.g. json.decoder.JSONDecodeError
            exception = e
        time.sleep(1)

    if numTotalResults == -1:
        print('Got the following exception when getting CPE count data via API: %s' % str(exception))
        return False

    # make necessary amount of API requests to pull all CPE data
    requestno = 0
    tasks = []
    while(offset <= numTotalResults):
        requestno += 1
        params = {'resultsPerPage': API_CPE_RESULTS_PER_PAGE, 'startIndex': offset}
        task = asyncio.ensure_future(worker(headers=headers, params=params, requestno=requestno, rate_limit=rate_limit))
        tasks.append(task)
        offset += API_CPE_RESULTS_PER_PAGE

    while True:
        finished_tasks, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED, timeout=2)
        if len(pending_tasks) < 1 or not UPDATE_SUCCESS:
            break

    if not UPDATE_SUCCESS:
        return False

    cpe_infos = []
    for task in finished_tasks:
        for cpe_triple in task.result()[0]:
            cpe_infos.append(cpe_triple)
    cpe_infos.sort(key=lambda cpe_info: cpe_info[0])

    # open CPE database and create tables
    if os.path.isfile(CPE_DATABASE_FILE):
        os.remove(CPE_DATABASE_FILE)
    db_conn = sqlite3.connect(CPE_DATABASE_FILE)
    db_cursor = db_conn.cursor()
    db_cursor.execute('''CREATE TABLE terms_to_entries (
                            term TEXT PRIMARY KEY,
                            entry_ids TEXT NOT NULL
                      );''')
    db_cursor.execute('''CREATE TABLE cpe_entries (
                            entry_id INTEGER PRIMARY KEY,
                            cpe TEXT,
                            term_frequencies TEXT,
                            abs_term_frequency REAL
                      );''')
    db_conn.commit()
    db_cursor.close()
    db_cursor = db_conn.cursor()

    # add CPE infos to DB
    terms_to_entries = {}
    for i, cpe_info in enumerate(cpe_infos):
        db_cursor.execute('INSERT INTO cpe_entries VALUES (?, ?, ?, ?)', (i, cpe_info[0], json.dumps(cpe_info[1]), cpe_info[2]))
        for term in cpe_info[1]:
            if term not in terms_to_entries:
                terms_to_entries[term] = []
            terms_to_entries[term].append(i)
    db_conn.commit()
    db_cursor.close()
    db_cursor = db_conn.cursor()

    # add term --> entries translations to DB
    for term, entry_ids in terms_to_entries.items():
        if not entry_ids:
            continue

        i = 1
        entry_ids_str = str(entry_ids[0])
        while i < len(entry_ids):
            start_i = i
            while (i < len(entry_ids) - 1) and entry_ids[i] + 1 == entry_ids[i+1]:
                i += 1
            if start_i == i:
                entry_ids_str += ',%d' % entry_ids[i]
            else:
                entry_ids_str += ',%d-%d' % (entry_ids[start_i], entry_ids[i])
            i += 1
        db_cursor.execute('INSERT INTO terms_to_entries VALUES (?, ?)', (term, entry_ids_str))

    db_conn.commit()
    db_cursor.close()
    db_conn.close()

    # create CPE deprecations file
    with open(DEPRECATED_CPES_FILE, "w") as outfile:
        final_deprecations = {}
        for task in finished_tasks:
            for deprecation in task.result()[1]:
                deprecated_cpe = list(deprecation)[0]
                if deprecated_cpe in final_deprecations:
                    final_deprecations[deprecated_cpe] = list(set(final_deprecations[deprecated_cpe] + deprecation[deprecated_cpe]))
                else:
                    final_deprecations[deprecated_cpe] = deprecation[deprecated_cpe]
        outfile.write('%s\n' % json.dumps(final_deprecations))
        outfile.close()

    return True


def _get_alternative_queries(init_queries):
    alt_queries_mapping = {}
    for query in init_queries:
        alt_queries_mapping[query] = []

        # replace 'httpd' with 'http' e.g. for Apache HTTP Server
        if 'httpd' in query:
            alt_query = query.replace('httpd', 'http')
            alt_queries_mapping[query].append(alt_query)

        # check for citrix 'adc' abbreviation
        if 'citrix' in query and (query.startswith('adc ') or query.endswith(' adc') or ' adc ' in query):
            alt_query = query.replace('adc', 'application delivery controller')
            alt_queries_mapping[query].append(alt_query)

        # check for dell 'omsa' abbreviation
        if 'dell' in query and (query.startswith('omsa ') or query.endswith(' omsa') or ' omsa ' in query):
            alt_query = query.replace('omsa', 'openmanage server administrator')
            alt_queries_mapping[query].append(alt_query)

        # fix bootstrap CPE naming
        if 'bootstrap' in query and 'getbootstrap' not in query:
            alt_queries_mapping[query].append(query + ' getbootstrap')

        # check for different variants of js library names, e.g. 'moment.js' vs. 'momentjs' vs. 'moment js'
        if 'js ' in query or ' js' in query or query.endswith('js'):
            words = query.split()
            alt_queries = []
            for i, word in enumerate(words):
                word = word.strip()
                new_query_words1, new_query_words2 = [], []
                if word == 'js' and i > 0:
                    new_query_words1 = words[:i-1] + [words[i-1] + 'js']
                    new_query_words2 = words[:i-1] + [words[i-1] + '.js']
                elif word.endswith('.js') or word.endswith('js'):
                    if i > 0:
                        new_query_words1 += words[:i]
                        new_query_words2 += words[:i]
                    if word.endswith('.js'):
                        new_query_words1 += [word[:-len('.js')]] + ['js']
                        new_query_words2 += [word[:-len('.js')] + 'js']
                    else:
                        new_query_words1 += [word[:-len('js')]] + ['js']
                        new_query_words2 += [word[:-len('js')] + '.js']

                if new_query_words1:
                    if i < len(words) - 1:
                        new_query_words1 += words[i+1:]
                        new_query_words2 += words[i+1:]
                    alt_queries.append(' '.join(new_query_words1))
                    alt_queries.append(' '.join(new_query_words2))

            if alt_queries:
                alt_queries_mapping[query] += alt_queries

        # split certain version parts with space, e.g. 'openssh 7.4p1' --> 'openssh 7.4 p1'
        pot_alt_query = ''
        cur_char_class = string.ascii_letters
        did_split, seen_first_break = False, False
        splits, maxsplit = 0, query.count(' ') + ALT_QUERY_MAXSPLIT
        for char in query:
            if char in (' ', '.', '-', '+'):
                seen_first_break = True
                pot_alt_query += char
                did_split = False
                continue

            if seen_first_break and splits < maxsplit and char not in cur_char_class and not did_split:
                pot_alt_query += ' '
                did_split = True
                splits += 1
                if char in string.ascii_letters:
                    cur_char_class = string.ascii_letters
                else:
                    try:
                        int(char)
                        cur_char_class = '0123456789'
                    except ValueError:
                        cur_char_class = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
            pot_alt_query += char

        # zero extend versions, e.g. 'Apache httpd 2.4' --> 'Apache httpd 2.4.0'
        version_match = VERSION_MATCH_ZE_RE.search(query)
        if version_match:
            alt_query = query.replace(version_match.group(0), version_match.group(0) + '.0')
            alt_queries_mapping[query].append(alt_query)

        pot_alt_query_parts = pot_alt_query.split()
        for i in range(len(pot_alt_query_parts)):
            if pot_alt_query_parts[i][-1] in ('.', '-', '+'):
                pot_alt_query_parts[i] = pot_alt_query_parts[i][:-1]
        pot_alt_query = ' '.join(pot_alt_query_parts)

        if pot_alt_query != query.strip():
            alt_queries_mapping[query].append(pot_alt_query)

    return alt_queries_mapping


def init_memdb():
    global DB_CONN_MEM

    if DB_CONN_MEM is None:
        DB_CONN_FILE = sqlite3.connect(CPE_DATABASE_FILE)
        DB_CONN_MEM = sqlite3.connect(DB_URI, uri=True)
        DB_CONN_FILE.backup(DB_CONN_MEM)
        DB_CONN_FILE.close()


def _search_cpes(queries_raw, count, threshold, keep_data_in_memory=False):
    """Facilitate CPE search as specified by the program arguments"""

    # create term frequencies and normalization factors for all queries
    queries = [query.lower() for query in queries_raw]

    # add alternative queries to improve retrieval
    alt_queries_mapping = _get_alternative_queries(queries)
    for alt_queries in alt_queries_mapping.values():
        queries += alt_queries

    query_infos = {}
    most_similar = {}
    all_query_words = set()
    for query in queries:
        query_tf = Counter(TEXT_TO_VECTOR_RE.findall(query))
        for term, tf in query_tf.items():
            query_tf[term] = tf / len(query_tf)
        all_query_words |= set(query_tf.keys())
        query_abs = math.sqrt(sum([cnt**2 for cnt in query_tf.values()]))
        query_infos[query] = (query_tf, query_abs)
        most_similar[query] = [("N/A", -1)]

    # set up DB connector
    if keep_data_in_memory:
        init_memdb()
        conn = sqlite3.connect(DB_URI, uri=True)
        db_cursor = conn.cursor()
    else:
        conn = sqlite3.connect(CPE_DATABASE_FILE, uri=True)
        db_cursor = conn.cursor()

    # figure out which CPE infos are relevant, based on the terms of all queries
    all_cpe_entry_ids = []
    for word in all_query_words:
        # query can only return one result, b/c term is PK
        db_query = 'SELECT entry_ids FROM terms_to_entries WHERE term = ?'
        cpe_entry_ids = db_cursor.execute(db_query, (word, )).fetchall()
        if not cpe_entry_ids or not cpe_entry_ids[0]:
            continue

        cpe_entry_ids = cpe_entry_ids[0][0].split(',')
        all_cpe_entry_ids.append(int(cpe_entry_ids[0]))

        for eid in cpe_entry_ids[1:]:
            if '-' in eid:
                eid = eid.split('-')
                all_cpe_entry_ids += list(range(int(eid[0]), int(eid[1])+1))
            else:
                all_cpe_entry_ids.append(int(eid))

    # iterate over all retrieved CPE infos and find best matching CPEs for queries
    iterator = []
    max_results_per_query = 250000
    remaining = len(all_cpe_entry_ids)
    is_one_iter_enough = remaining <= max_results_per_query
    while remaining > 0:
        if remaining > max_results_per_query:
            count_params_in_str = max_results_per_query
        else:
            count_params_in_str = remaining
        param_in_str = ('?,' * count_params_in_str)[:-1]
        if keep_data_in_memory or not is_one_iter_enough:
            db_query = 'SELECT cpe, term_frequencies, abs_term_frequency FROM cpe_entries WHERE entry_id IN (%s)' % param_in_str
            cpe_infos = db_cursor.execute(db_query, all_cpe_entry_ids[remaining-count_params_in_str:remaining]).fetchall()
            iterator += cpe_infos
        else:
            db_query = 'SELECT cpe, term_frequencies, abs_term_frequency FROM cpe_entries WHERE entry_id IN (%s)' % param_in_str
            db_cursor.execute(db_query, all_cpe_entry_ids[remaining-count_params_in_str:remaining])
            iterator = db_cursor

        remaining -= max_results_per_query

    for cpe_info in iterator:
        cpe, cpe_tf, cpe_abs = cpe_info
        cpe_tf = json.loads(cpe_tf)
        cpe_abs = float(cpe_abs)

        for query in queries:
            query_tf, query_abs = query_infos[query]
            intersecting_words = set(cpe_tf.keys()) & set(query_tf.keys())
            inner_product = sum([cpe_tf[w] * query_tf[w] for w in intersecting_words])

            normalization_factor = cpe_abs * query_abs

            if not normalization_factor:  # avoid divison by 0
                continue

            sim_score = float(inner_product)/float(normalization_factor)

            if threshold > 0 and sim_score < threshold:
                continue

            cpe_base = ':'.join(cpe.split(':')[:5]) + ':'
            if sim_score > most_similar[query][0][1]:
                most_similar[query] = [(cpe, sim_score)] + most_similar[query][:count-1]
            elif not most_similar[query][0][0].startswith(cpe_base):
                insert_idx = None
                for i, (cur_cpe, cur_sim_score) in enumerate(most_similar[query][1:]):
                    if sim_score > cur_sim_score:
                        if not cur_cpe.startswith(cpe_base):
                            insert_idx = i+1
                        break
                if insert_idx:
                    if len(most_similar[query]) < count:
                        most_similar[query] = most_similar[query][:insert_idx] + [(cpe, sim_score)] + most_similar[query][insert_idx:]
                    else:
                        most_similar[query] = most_similar[query][:insert_idx] + [(cpe, sim_score)] + most_similar[query][insert_idx:-1]

    # create intermediate results (including any additional queries)
    intermediate_results = {}
    for query in queries:
        if most_similar[query] and len(most_similar[query]) == 1 and most_similar[query][0][1] == -1:
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

        if query not in intermediate_results and (query not in alt_queries_mapping or not alt_queries_mapping[query]):
            continue

        if query not in alt_queries_mapping or not alt_queries_mapping[query]:
            results[query_raw] = intermediate_results[query]
        else:
            most_similar = None
            if query in intermediate_results:
                most_similar = intermediate_results[query]
            for alt_query in alt_queries_mapping[query]:
                if alt_query not in intermediate_results:
                    continue
                if not most_similar or intermediate_results[alt_query][0][1] > most_similar[0][1]:
                    most_similar = intermediate_results[alt_query]
            results[query_raw] = most_similar

    return results


def is_cpe_equal(cpe1, cpe2):
    """Return True if both CPEs are considered equal, False otherwise"""

    if len(cpe1) != len(cpe2):
        return False

    for i in range(len(cpe1)):
        if cpe1[i] != cpe2[i]:
            if not(cpe1[i] in ('*', '-') and cpe2[i] in('*', '-')):
                return False
    return True


def match_cpe23_to_cpe23_from_dict(cpe23_in, keep_data_in_memory=False):
    """
    Try to return a valid CPE 2.3 string that exists in the NVD's CPE
    dictionary based on the given, potentially badly formed, CPE string.
    """

    # if the given CPE is simply not a full CPE 2.3 string
    pot_new_cpe = ''
    if cpe23_in.count(':') < 12:
        pot_new_cpe = cpe23_in
        if pot_new_cpe.endswith(':'):
            pot_new_cpe += '*'
        while pot_new_cpe.count(':') < 12:
            pot_new_cpe += ':*'

    pre_cpe_in = cpe23_in
    while pre_cpe_in.count(':') > 3:  # break if next cpe part would be vendor part
        pre_cpe_in = pre_cpe_in[:-1]
        if pre_cpe_in.endswith(':') or pre_cpe_in.count(':') > 9:  # skip rear parts in fixing process
            continue

        all_cpes = get_all_cpes(keep_data_in_memory)
        for cpe in all_cpes:
            if cpe23_in == cpe:
                return cpe23_in
            if pot_new_cpe and pot_new_cpe == cpe:
                return pot_new_cpe

            if pre_cpe_in in cpe:
                # stitch together the found prefix and the remaining part of the original CPE
                if cpe23_in[len(pre_cpe_in)] == ':':
                    cpe_in_add_back = cpe23_in[len(pre_cpe_in)+1:]
                else:
                    cpe_in_add_back = cpe23_in[len(pre_cpe_in):]
                new_cpe = '%s:%s' % (pre_cpe_in, cpe_in_add_back)

                # get new_cpe to full CPE 2.3 length by adding or removing wildcards
                while new_cpe.count(':') < 12:
                    new_cpe += ':*'
                if new_cpe.count(':') > 12:
                    new_cpe = new_cpe[:new_cpe.rfind(':')]

                # if a matching CPE was found, return it
                if is_cpe_equal(new_cpe, cpe):
                    return cpe
    return ''


def create_cpe_from_base_cpe_and_query(cpe, query):
    version_str_match = VERSION_MATCH_CPE_CREATION_RE.search(query)
    if version_str_match:
        version_str = version_str_match.group(1).strip()

        # always put version into the appropriate, i.e. sixth, CPE field
        cpe_parts = cpe.split(':')
        cpe_parts[5] = version_str
        return ':'.join(cpe_parts)

    return None


def is_versionless_query(query):
    version_str_match = VERSION_MATCH_CPE_CREATION_RE.search(query)
    if not version_str_match:
        return True
    return False


def create_base_cpe_if_versionless_query(cpe, query):
    if is_versionless_query(query):
        cpe_parts = cpe.split(':')
        base_cpe = ':'.join(cpe_parts[:5] + ['*'] * 8)
        return base_cpe

    return None


def get_all_cpes(keep_data_in_memory=False):
    if keep_data_in_memory:
        init_memdb()
        conn = sqlite3.connect(DB_URI, uri=True)
        db_cursor = conn.cursor()
    else:
        conn = sqlite3.connect(CPE_DATABASE_FILE, uri=True)
        db_cursor = conn.cursor()

    cpes = db_cursor.execute('SELECT cpe FROM cpe_entries').fetchall()
    cpes = [cpe[0] for cpe in cpes]

    return cpes


def search_cpes(queries, count=3, threshold=-1, keep_data_in_memory=False):
    if not queries:
        return {}

    if isinstance(queries, str):
        queries = [queries]

    return _search_cpes(queries, count, threshold, keep_data_in_memory)


if __name__ == "__main__":
    SILENT = not sys.stdout.isatty()
    args = parse_args()

    if args.verbose:
        SILENT = False

    perform_update = False

    if args.update:
        perform_update = True

    if args.queries and not os.path.isfile(CPE_DATABASE_FILE):
        if not SILENT:
            print("[+] Running initial setup (might take a couple of minutes)", file=sys.stderr)
        perform_update = True

    if perform_update:
        import asyncio
        import aiohttp
        from aiolimiter import AsyncLimiter
        import requests

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update(args.api_key))

        if not UPDATE_SUCCESS:
            print('[-] Failed updating the local CPE database!')

    if args.queries:
        results = search_cpes(args.queries, args.count)

        if not results:
            print()
            print(results)

        for i, query in enumerate(results):
            if not SILENT and i > 0:
                print()

            print(results[query][0][0])
            if not SILENT:
                pprint.pprint(results[query])
