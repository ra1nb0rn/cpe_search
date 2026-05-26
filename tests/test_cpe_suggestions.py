#!/usr/bin/env python3

import os
import unittest

from cpe_search.cpe_search import search_cpes


class TestSearches(unittest.TestCase):

    def test_search_wp_100_42_3(self):
        self.maxDiff = None
        query = "WordPress 100.42.3"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:wordpress:wordpress:100.42.3:*:*:*:*:*:*:*", -0.7149856309531066),
            ("cpe:2.3:a:wordpress:wordpress:-:*:*:*:*:*:*:*", 0.7149856309531066),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_apache_airflow_100_42_3(self):
        self.maxDiff = None
        query = "Airflow 100.42.3"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:apache:airflow:100.42.3:*:*:*:*:*:*:*", -0.4463714088254057),
            ("cpe:2.3:a:apache:airflow:-:*:*:*:*:*:*:*", 0.4463714088254057),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_apache_airflow_no_version(self):
        self.maxDiff = None
        query = "Airflow"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:apache:airflow:*:*:*:*:*:*:*:*", 0.6277465574883976)
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_jquery_100_42_3(self):
        self.maxDiff = None
        query = "jQuery 100.42.3"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:jquery:jquery:100.42.3:*:*:*:*:*:*:*", -0.7400648268105541),
            ("cpe:2.3:a:jquery:jquery:-:*:*:*:*:*:*:*", 0.7400648268105541),
            ("cpe:2.3:a:jquery:jquery:1.0.1:*:*:*:*:*:*:*", 0.6410875586846989),
            ("cpe:2.3:a:jquery:jquery_ui:100.42.3:*:*:*:*:*:*:*", -0.6115615996256556),
            ("cpe:2.3:a:jquery:jquery_ui:1.10.0:rc1:*:*:*:*:*:*", 0.6115615996256556),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_jfrog_artifactory_4_29_0(self):
        self.maxDiff = None
        query = "jfrog artifactory 4.29.0"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:jfrog:artifactory:4.29.0:*:*:*:*:*:*:*", -0.8389898738252954),
            ("cpe:2.3:a:jfrog:artifactory:-:*:*:*:*:-:*:*", 0.8389898738252954),
            ("cpe:2.3:a:jfrog:artifactory:1.3.0:-:*:*:*:-:*:*", 0.7156686848066457),
            ("cpe:2.3:a:jfrog:artifactory:1.3.0:beta3:*:*:*:-:*:*", 0.6635959168415716),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_dell_omsa_9_4_0_2(self):
        self.maxDiff = None
        query = "dell omsa 9.4.0.2"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            (
                "cpe:2.3:a:dell:openmanage_server_administrator:9.4.0.2:*:*:*:*:*:*:*",
                -0.8598904081377858,
            ),
            (
                "cpe:2.3:a:dell:openmanage_server_administrator:-:*:*:*:*:*:*:*",
                0.8598904081377858,
            ),
            (
                "cpe:2.3:a:dell:openmanage_server_administrator:5.2.0:*:*:*:*:*:*:*",
                0.842705567846005,
            ),
            (
                "cpe:2.3:a:dell:openmanage_server_administrator_installer:9.4.0.2:*:*:*:*:*:*:*",
                -0.7520362114046224,
            ),
            (
                "cpe:2.3:a:dell:openmanage_server_administrator_installer:1.0.0:*:*:*:*:*:*:*",
                0.7520362114046224,
            ),
            (
                "cpe:2.3:a:dell:openmanage_server_administrator_lite:9.4.0.2:*:*:*:*:*:*:*",
                -0.7520362114046224,
            ),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_citrix_adc_13_1_42_47(self):
        self.maxDiff = None
        query = "citrix adc 13.1-42.47"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            (
                "cpe:2.3:a:citrix:application_delivery_controller:13.1:42.47:*:*:-:*:*:*",
                -0.9203872821293977,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:13.1-42.47:*:*:*:*:*:*:*",
                -0.9203872821293977,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:13.1:*:*:*:-:*:*:*",
                0.9203872821293977,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:13.1:42.47:*:*:fips:*:*:*",
                -0.8761666840502798,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:13.1:*:*:*:fips:*:*:*",
                0.8761666840502798,
            ),
            (
                "cpe:2.3:a:citrix:netscaler_application_delivery_controller:13.1:42.47:*:*:-:*:*:*",
                -0.8368272939125518,
            ),
            (
                "cpe:2.3:a:citrix:netscaler_application_delivery_controller:13.1-42.47:*:*:*:*:*:*:*",
                -0.8368272939125518,
            ),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_citrix_adc_no_version(self):
        self.maxDiff = None
        query = "citrix adc"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            (
                "cpe:2.3:h:citrix:application_delivery_controller:*:*:*:*:*:*:*:*",
                0.937628771015901,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:*:*:*:*:-:*:*:*",
                0.8807837026194261,
            ),
            (
                "cpe:2.3:o:citrix:application_delivery_controller_firmware:*:*:*:*:*:*:*:*",
                0.8454539420916392,
            ),
            (
                "cpe:2.3:h:citrix:netscaler_application_delivery_controller:*:*:*:*:*:*:*:*",
                0.8398726044979914,
            ),
            (
                "cpe:2.3:a:citrix:application_delivery_controller:*:*:*:*:fips:*:*:*",
                0.836591883535865,
            ),
            (
                "cpe:2.3:a:citrix:netscaler_application_delivery_controller:*:*:*:*:*:*:*:*",
                0.802816823924362,
            ),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_openssh_83_p4(self):
        self.maxDiff = None
        query = "openssh 8.3 p4"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            ("cpe:2.3:a:openbsd:openssh:8.3:p4:*:*:*:*:*:*", -0.7329519711504404),
            ("cpe:2.3:a:openbsd:openssh:8.3_p4:*:*:*:*:*:*:*", -0.7329519711504404),
            ("cpe:2.3:a:openbsd:openssh:8.3:*:*:*:*:*:*:*", 0.7329519711504404),
            ("cpe:2.3:a:openssh:openssh:8.3:*:*:*:*:*:*:*", -0.6719429387690861),
            ("cpe:2.3:a:openssh:openssh:8.3:p4:*:*:*:*:*:*", -0.6719429387690861),
            ("cpe:2.3:a:openssh:openssh:8.3_p4:*:*:*:*:*:*:*", -0.6719429387690861),
            ("cpe:2.3:a:openssh:openssh:-:*:*:*:*:*:*:*", 0.6719429387690861),
            ("cpe:2.3:a:openbsd:openssh:8.3:p1:*:*:*:*:*:*", 0.6684082619040745),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_datatables_1_9_4(self):
        self.maxDiff = None
        query = "datatables 1.9.4"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            (
                "cpe:2.3:a:sprymedia:datatables:1.9.4:*:*:*:*:*:*:*",
                -0.4405465471359274,
            ),
            ("cpe:2.3:a:sprymedia:datatables:1.9.2:*:*:*:*:jquery:*:*", 0.4405465471359274),
            ("cpe:2.3:a:datatables:datatables.net:1.9.4:*:*:*:*:*:*:*", -0.4205754118072099),
            (
                "cpe:2.3:a:datatables:datatables.net:1.10.0:-:*:*:*:node.js:*:*",
                0.4205754118072099,
            ),
            (
                "cpe:2.3:a:datatables:datatables.net:1.10.0:beta1:*:*:*:node.js:*:*",
                0.4050687916768836,
            ),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])

    def test_search_microsoft_sql_server_2019_15_00_2000_00_RTM(self):
        self.maxDiff = None
        query = "microsoft sql_server 2019 15.00.2000.00;RTM"
        result = search_cpes(query)["pot_cpes"]
        expected_related_cpes = [
            (
                "cpe:2.3:a:microsoft:sql_server:2019:15.00.2000.00:*:*:*:*:*:*",
                -0.7346392073827277,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:2019:15.00.2000.00:RTM:*:*:*:*:*",
                -0.7346392073827277,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:2019_15.00.2000.00_RTM:*:*:*:*:*:*:*",
                -0.7346392073827277,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:2019:*:*:*:*:*:*:*",
                0.7346392073827277,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server_2019:2019:*:*:*:*:*:*:*",
                -0.6996692642461084,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server_2019:2019:15.00.2000.00:*:*:*:*:*:*",
                -0.6996692642461084,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server_2019:2019:15.00.2000.00:RTM:*:*:*:*:*",
                -0.6996692642461084,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server_2019:2019_15.00.2000.00_RTM:*:*:*:*:*:*:*",
                -0.6996692642461084,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server_2019:15.0.2000.5:*:*:*:*:*:*:*",
                0.6996692642461084,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:-:*:*:*:*:*:*:*",
                0.6940097229719058,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:2019:15.00.2000.00:*:*:*:*:x64:*",
                -0.6927881041154651,
            ),
            (
                "cpe:2.3:a:microsoft:sql_server:2019:15.00.2000.00:RTM:*:*:*:x64:*",
                -0.6927881041154651,
            ),
            ("cpe:2.3:a:microsoft:sql_server:2019:*:*:*:*:*:x64:*", 0.6927881041154651),
        ]
        for i, (expected_related_cpe, match_score) in enumerate(expected_related_cpes):
            self.assertEqual(expected_related_cpe, result[i][0])
            self.assertAlmostEqual(match_score, result[i][1])


if __name__ == "__main__":
    os.environ["IS_CPE_SEARCH_TEST"] = "true"
    unittest.main()
