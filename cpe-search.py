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
CPE_DATA_FILES = {"2.2": os.path.join(SCRIPT_DIR, "cpe-search-dictionary_v2.2.json"),
                  "2.3": os.path.join(SCRIPT_DIR, "cpe-search-dictionary_v2.3.json")}
CPE_DICT_ITEM_RE = re.compile(r"<cpe-item name=\"([^\"]*)\">\s*\n\s*<title[^>]*>([^<]*)</title>.+?(?=<cpe-23:)<cpe-23:cpe23-item name=\"([^\"]*)\"", flags=re.DOTALL)
TEXT_TO_VECTOR_RE = re.compile(r"[\w+\.]+")
CPE_INFOS = []


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)")
    parser.add_argument("-u", "--update", action="store_true", help="Update the local CPE database")
    parser.add_argument("-c", "--count", default=3, type=int, help="The number of CPEs to show in the similarity overview (default: 3)")
    parser.add_argument("-v", "--version", default="2.2", choices=["2.2", "2.3"], help="The CPE version to use: 2.2 or 2.3 (default: 2.2)")
    parser.add_argument("-q", "--query", dest="queries", metavar="QUERY", action="append", help="A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'")
    # parser.add_argument("queries", metavar="QUERY", nargs="+", help="A query, i.e. textual software info like 'Apache 2.4.39' or 'Wordpress 5.7.2'")

    args = parser.parse_args()
    if not args.update and not args.queries:
        parser.print_help()
    return args


def update(cpe_version):
    """Update locally stored CPE information"""

    # download dictionary
    if sys.stdout.isatty():
        print("[+] Downloading NVD's official CPE dictionary (might take some time)")
    src = CPE_DICT_URL
    dst = os.path.join(SCRIPT_DIR, src.rsplit("/", 1)[1])
    urlretrieve(CPE_DICT_URL, dst)

    # unzip CPE dictionary
    if sys.stdout.isatty():
        print("[+] Unzipping dictionary")
    with zipfile.ZipFile(dst,"r") as zip_ref:
        cpe_dict_name = zip_ref.namelist()[0]
        cpe_dict_filepath = os.path.join(SCRIPT_DIR, cpe_dict_name)
        zip_ref.extractall(SCRIPT_DIR)

    # build custom CPE database, additionally containing term frequencies and normalization factors
    if sys.stdout.isatty():
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
    with open(CPE_DATA_FILES["2.2"], "w") as fout:
        json.dump(cpe22_infos, fout)
    with open(CPE_DATA_FILES["2.3"], "w") as fout:
        json.dump(cpe23_infos, fout)

    # clean up
    if sys.stdout.isatty():
        print("[+] Cleaning up")
    os.remove(dst)
    os.remove(os.path.join(SCRIPT_DIR, cpe_dict_name))

    # set CPE infos
    global CPE_INFOS
    if cpe_version == "2.2":
        CPE_INFOS = cpe22_infos
    else:
        CPE_INFOS = cpe23_infos


def search_cpes(args):
    """Facilitate CPE search as specified by the program arguments"""

    # create term frequencies and normalization factors for all queries
    queries = [query.lower() for query in args.queries]
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
    for cpe, cpe_tf, cpe_abs in CPE_INFOS:
        for query in queries:
            query_tf, query_abs = query_infos[query]
            intersecting_words = set(cpe_tf.keys()) & set(query_tf.keys())
            inner_product = sum([cpe_tf[w] * query_tf[w] for w in intersecting_words])

            normalization_factor = cpe_abs * query_abs

            if not normalization_factor:  # avoid divison by 0
                continue

            sim_score = float(inner_product)/float(normalization_factor)
            if sim_score > most_similar[query][0][1]:
                most_similar[query] = [(cpe, sim_score)] + most_similar[query][:args.count-1]
            elif len(most_similar[query]) < args.count:
                most_similar[query].append((cpe, sim_score))

    # print results
    for i, query in enumerate(args.queries):
        if i > 0:
            print()
        print(most_similar[query.lower()][0][0])
        if sys.stdout.isatty():
            pprint.pprint(most_similar[query.lower()])

# main
args = parse_args()
if args.update:
    update(args.version)

if args.queries and not CPE_INFOS and not os.path.isfile(CPE_DATA_FILES[args.version]):
    print("[+] Running initial setup (might take a couple of minutes)", file=sys.stderr)
    update(args.version)

if args.queries:
    if not CPE_INFOS:
        with open(CPE_DATA_FILES[args.version], "r") as f:
            CPE_INFOS = json.load(f)
    search_cpes(args)
