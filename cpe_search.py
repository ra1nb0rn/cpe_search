#!/usr/bin/env python3

import argparse
from collections import Counter
import math
import os
import pprint
import re
import sys
from urllib.parse import unquote
from urllib.request import urlretrieve
import zipfile

try:  # use ujson if available
    import ujson as json
except ModuleNotFoundError:
    import json

# Constants
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CPE_DICT_URL = "https://nvd.nist.gov/feeds/xml/cpe/dictionary/official-cpe-dictionary_v2.3.xml.zip"
CPE_DATA_FILES = {"2.2": os.path.join(SCRIPT_DIR, "cpe-search-dictionary_v2.2.csv"),
                  "2.3": os.path.join(SCRIPT_DIR, "cpe-search-dictionary_v2.3.csv")}
CPE_DICT_ITEM_RE = re.compile(r"<cpe-item name=\"([^\"]*)\">.*?<title xml:lang=\"en-US\"[^>]*>([^<]*)</title>.*?<cpe-23:cpe23-item name=\"([^\"]*)\"", flags=re.DOTALL)
TEXT_TO_VECTOR_RE = re.compile(r"[\w+\.]+")
GET_ALL_CPES_RE = re.compile(r'(.*);.*;.*')
ALL_CPES = []
SILENT = True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)")
    parser.add_argument("-u", "--update", action="store_true", help="Update the local CPE database")
    parser.add_argument("-c", "--count", default=3, type=int, help="The number of CPEs to show in the similarity overview (default: 3)")
    parser.add_argument("-v", "--version", default="2.2", choices=["2.2", "2.3"], help="The CPE version to use: 2.2 or 2.3 (default: 2.2)")
    parser.add_argument("-q", "--query", dest="queries", metavar="QUERY", action="append", help="A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'")

    args = parser.parse_args()
    if not args.update and not args.queries:
        parser.print_help()
    return args


def set_silent(silent):
    global SILENT
    SILENT = silent


def update(cpe_version):
    """Update locally stored CPE information"""

    # download dictionary
    if not SILENT:
        print("[+] Downloading NVD's official CPE dictionary (might take some time)")
    src = CPE_DICT_URL
    dst = os.path.join(SCRIPT_DIR, src.rsplit("/", 1)[1])
    urlretrieve(CPE_DICT_URL, dst)

    # unzip CPE dictionary
    if not SILENT:
        print("[+] Unzipping dictionary")
    with zipfile.ZipFile(dst,"r") as zip_ref:
        cpe_dict_name = zip_ref.namelist()[0]
        cpe_dict_filepath = os.path.join(SCRIPT_DIR, cpe_dict_name)
        zip_ref.extractall(SCRIPT_DIR)

    # build custom CPE database, additionally containing term frequencies and normalization factors
    if not SILENT:
        print("[+] Creating a custom CPE database for future invocations")
    cpe22_infos, cpe23_infos = [], []
    with open(cpe_dict_filepath) as fin:
        content = fin.read()
        cpe_items = CPE_DICT_ITEM_RE.findall(content)
        for cpe_item in cpe_items:
            cpe22, cpe_name, cpe23 = cpe_item[0].lower(), cpe_item[1].lower(), cpe_item[-1].lower()

            for i, cpe in enumerate((cpe22, cpe23)):
                if "%" in cpe:
                    cpe = unquote(cpe)
                cpe_mod = cpe.replace("_", ":")

                if i == 0:
                    cpe_elems = cpe_mod[7:].split(":")
                else:
                    cpe_elems = cpe_mod[10:].split(":")

                cpe_name_elems = cpe_name.split()
                words = TEXT_TO_VECTOR_RE.findall(" ".join(cpe_elems + cpe_name_elems))
                cpe_tf = Counter(words)
                for term, tf in cpe_tf.items():
                    cpe_tf[term] = tf / len(cpe_tf)
                cpe_abs = math.sqrt(sum([cnt**2 for cnt in cpe_tf.values()]))

                cpe_info = (cpe, cpe_tf, cpe_abs)
                if i == 0:
                    cpe22_infos.append(cpe_info)
                else:
                    cpe23_infos.append(cpe_info)

    # store customly built CPE database
    if cpe_version == "2.2":
        with open(CPE_DATA_FILES["2.2"], "w") as fout:
            for cpe, cpe_tf, cpe_abs in cpe22_infos:
                fout.write('%s;%s;%f\n' % (cpe, json.dumps(cpe_tf), cpe_abs))
    else:
        with open(CPE_DATA_FILES["2.3"], "w") as fout:
            for cpe, cpe_tf, cpe_abs in cpe23_infos:
                fout.write('%s;%s;%f\n' % (cpe, json.dumps(cpe_tf), cpe_abs))

    # clean up
    if not SILENT:
        print("[+] Cleaning up")
    os.remove(dst)
    os.remove(os.path.join(SCRIPT_DIR, cpe_dict_name))


def _search_cpes(queries_raw, cpe_version, count, threshold):
    """Facilitate CPE search as specified by the program arguments"""

    # create term frequencies and normalization factors for all queries
    queries = [query.lower() for query in queries_raw]
    query_infos = {}
    most_similar = {}
    for query in queries:
        query_tf = Counter(TEXT_TO_VECTOR_RE.findall(query))
        for term, tf in query_tf.items():
            query_tf[term] = tf / len(query_tf)
        query_abs = math.sqrt(sum([cnt**2 for cnt in query_tf.values()]))
        query_infos[query] = (query_tf, query_abs)
        most_similar[query] = [("N/A", -1)]

    # iterate over every CPE, for every query compute similarity scores and keep track of most similar CPEs
    with open(CPE_DATA_FILES[cpe_version], "r") as fout:
        for line in fout:
            cpe, cpe_tf, cpe_abs = line.rsplit(';', maxsplit=2)
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

                if sim_score > most_similar[query][0][1]:
                    most_similar[query] = [(cpe, sim_score)] + most_similar[query][:count-1]
                elif len(most_similar[query]) < count:
                    most_similar[query].append((cpe, sim_score))

    # create results
    results = {}
    for query in queries_raw:
        results[query] = most_similar[query.lower()]

        rm_idxs = []
        for i, result in enumerate(results[query]):
            if result[1] == -1:
                rm_idxs.append(i)

        for i in rm_idxs:
            del results[query][i]

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


def _match_cpe23_to_cpe23_from_dict_memory(cpe23_in, keep_data_in_memory=False):
    """
    Try to return a valid CPE 2.3 string that exists in the NVD's CPE
    dictionary based on the given, potentially badly formed, CPE string.
    """

    global ALL_CPES

    if not ALL_CPES:
        all_cpes = get_all_cpes(version='2.3')
        ALL_CPES = all_cpes
    else:
        all_cpes = ALL_CPES

    # if CPE is already in the NVD dictionary
    if cpe23_in in all_cpes:
        return cpe23_in

    # if the given CPE is simply not a full CPE 2.3 string
    pot_new_cpe = ''
    if cpe23_in.count(':') < 12:
        pot_new_cpe = cpe23_in
        if pot_new_cpe.endswith(':'):
            pot_new_cpe += '*'
        while pot_new_cpe.count(':') < 12:
            pot_new_cpe += ':*'

    # if the given CPE is simply not a full CPE 2.3 string
    if cpe23_in.count(':') < 12:
        new_cpe = cpe23_in
        if new_cpe.endswith(':'):
            new_cpe += '*'
        while new_cpe.count(':') < 12:
            new_cpe += ':*'
        for pot_cpe in all_cpes:
            if new_cpe == pot_cpe:
                return pot_cpe

    # try to "fix" badly formed CPE strings like
    # "cpe:2.3:a:proftpd:proftpd:1.3.3c:..." vs. "cpe:2.3:a:proftpd:proftpd:1.3.3:c:..."
    pre_cpe_in = cpe23_in
    while pre_cpe_in.count(':') > 3:  # break if next cpe part would be vendor part
        pre_cpe_in = pre_cpe_in[:-1]
        if pre_cpe_in.endswith(':') or pre_cpe_in.count(':') > 9:  # skip rear parts in fixing process
            continue

        for pot_cpe in all_cpes:
            if pre_cpe_in in pot_cpe:

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
                if is_cpe_equal(new_cpe, pot_cpe):
                    return pot_cpe

    return ''


def _match_cpe23_to_cpe23_from_dict_file(cpe23_in, keep_data_in_memory=False):
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

        with open(CPE_DATA_FILES['2.3'], "r") as fout:
            for line in fout:
                cpe = line.rsplit(';', maxsplit=2)[0].strip()

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


def match_cpe23_to_cpe23_from_dict(cpe23_in, keep_data_in_memory=False):
    """
    Try to return a valid CPE 2.3 string that exists in the NVD's CPE
    dictionary based on the given, potentially badly formed, CPE string.
    """

    if not keep_data_in_memory:
        return _match_cpe23_to_cpe23_from_dict_file(cpe23_in)
    else:
        return _match_cpe23_to_cpe23_from_dict_memory(cpe23_in)


def get_all_cpes(version):
    with open(CPE_DATA_FILES[version], "r") as f:
        cpes = GET_ALL_CPES_RE.findall(f.read())
    return cpes


def search_cpes(queries, cpe_version="2.3", count=3, threshold=-1):
    if not queries:
        return {}

    if isinstance(queries, str):
        queries = [queries]

    return _search_cpes(queries, cpe_version, count, threshold)


if __name__ == "__main__":
    SILENT = not sys.stdout.isatty()
    args = parse_args()
    if args.update:
        update(args.version)

    if args.queries and not os.path.isfile(CPE_DATA_FILES[args.version]):
        if not SILENT:
            print("[+] Running initial setup (might take a couple of minutes)", file=sys.stderr)
        update(args.version)

    if args.queries:
        results = search_cpes(args.queries, args.version, args.count)

        for i, query in enumerate(results):
            if not SILENT and i > 0:
                print()

            print(results[query][0][0])
            if not SILENT:
                pprint.pprint(results[query])
