#!/usr/bin/env python3

import unittest
from cpe_search import search_cpes

class TestSearches(unittest.TestCase):

    def test_search_wp_572(self):
        self.maxDiff = None
        query = 'WordPress 5.7.2'
        test_best_match_cpe = 'cpe:2.3:a:wordpress:wordpress:5.7.2:*:*:*:*:*:*:*'
        test_best_match_score = '0.98058067569092'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_apache_2425(self):
        self.maxDiff = None
        query = 'Apache 2.4.25'
        test_best_match_cpe = 'cpe:2.3:a:apache:http_server:2.4.25:*:*:*:*:*:*:*'
        test_best_match_score = '0.7372097807744855'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_proftpd_133c(self):
        self.maxDiff = None
        query = 'Proftpd 1.3.3c'
        test_best_match_cpe = 'cpe:2.3:a:proftpd:proftpd:1.3.3:c:*:*:*:*:*:*'
        test_best_match_score = '0.8333333333333334'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_thingsboard_341(self):
        self.maxDiff = None
        query = 'Thingsboard 3.4.1'
        test_best_match_cpe = 'cpe:2.3:a:thingsboard:thingsboard:3.4.1:*:*:*:*:*:*:*'
        test_best_match_score = '0.98058067569092'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_redis_323(self):
        self.maxDiff = None
        query = 'Redis 3.2.3'
        test_best_match_cpe = 'cpe:2.3:a:redis:redis:3.2.3:*:*:*:*:*:*:*'
        test_best_match_score = '0.98058067569092'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_piwik_045(self):
        self.maxDiff = None
        query = 'Piwik 0.4.5'
        test_best_match_cpe = 'cpe:2.3:a:piwik:piwik:0.4.5:*:*:*:*:*:*:*'
        test_best_match_score = '0.98058067569092'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_vmware_spring_framework_5326(self):
        self.maxDiff = None
        query = 'VMWare Spring Framework 5.3.26'
        test_best_match_cpe = 'cpe:2.3:a:vmware:spring_framework:5.3.26:*:*:*:*:*:*:*'
        test_best_match_score = '1.0'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_zulip_48(self):
        self.maxDiff = None
        query = 'Zulip 4.8'
        test_best_match_cpe = 'cpe:2.3:a:zulip:zulip:4.8:*:*:*:*:*:*:*'
        test_best_match_score = '0.98058067569092'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

    def test_search_electron_1317(self):
        self.maxDiff = None
        query = 'Electron 13.1.7'
        test_best_match_cpe = 'cpe:2.3:a:electronjs:electron:13.1.7:*:*:*:*:*:*:*'
        test_best_match_score = '0.816496580927726'
        result = search_cpes(queries=[query])
        self.assertEqual(result[query][0][0], test_best_match_cpe)
        self.assertEqual(str(result[query][0][1]), test_best_match_score)

if __name__ == '__main__':
    unittest.main()
