# cpe_search
Search for Common Platform Enumeration (CPE) strings using software names and titles.

## About
*cpe_search* can be used to search for Common Platform Enumeration (CPE) strings using software names and titles. For example, if some tool discovered a web server running *Apache 2.4.39*, you can use this tool to easily and quickly retrieve the corresponding CPE string *cpe:2.3:a:apache:http_server:2.4.39:*:*:*:*:*:*:**. Thereafter, the retrieved CPE string can be used to accurately search for vulnerabilities, e.g. via the [Online NVD](https://nvd.nist.gov/) or [AVAIN](https://github.com/ra1nb0rn/avain)'s *avain-cve_correlation* subtool. *cpe_search* supports CPE 2.3.

## Usage
*cpe_search*'s usage information is shown in the following:
```
usage: cpe_search.py [-h] [-u] [-c COUNT] [-q QUERY]

Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)

optional arguments:
  -h, --help            show this help message and exit
  -u, --update          Update the local CPE database
  -c COUNT, --count COUNT
                        The number of CPEs to show in the similarity overview (default: 3)
  -q QUERY, --query QUERY
                        A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'
```
Note that when querying software with ``-q`` you have to put the software information in quotes if it contains any spaces. Also, you can use ``-q`` multiple times to make multiple queries at once. Moreover, the output can be piped to be directly useable with other tools. Here are some examples:
* Query *Sudo 1.8.2* to retrieve its CPE 2.3 string:
  ```bash
  $ ./cpe_search.py -q "Sudo 1.8.2"
  cpe:2.3:a:sudo_project:sudo:1.8.2:*:*:*:*:*:*:*
  [('cpe:2.3:a:sudo_project:sudo:1.8.2:*:*:*:*:*:*:*', 0.8660254037844385),
   ('cpe:2.3:a:sudo_project:sudo:1.3.0:*:*:*:*:*:*:*', 0.5773502691896256),
   ('cpe:2.3:a:cryptography.io:cryptography:1.8.2:*:*:*:*:*:*:*',
    0.4714045207910316)]
  ```
* Make a query and pipe the retrieved CPE to another tool:
  ```bash
  $ ./cpe_search.py -q "Windows 10 1809" | xargs echo
  cpe:2.3:o:microsoft:windows_10:1809:*:*:*:*:*:*:*
  ```
* Make two queries at once:
  ```bash
  $ ./cpe_search.py -q "Apache 2.4.39" -q "Wordpress 5.7.2"
  cpe:2.3:a:apache:http_server:2.4.39:*:*:*:*:*:*:*
  [('cpe:2.3:a:apache:http_server:2.4.39:*:*:*:*:*:*:*', 0.6666664603674289),
  ('cpe:2.3:a:apache:apache-airflow-providers-apache-spark:-:*:*:*:*:*:*:*',
    0.600000153741923),
  ('cpe:2.3:a:apache:apache-airflow-providers-apache-hive:-:*:*:*:*:*:*:*',
    0.600000153741923)]

  cpe:2.3:a:wordpress:wordpress:5.7.2:*:*:*:*:*:*:*
  [('cpe:2.3:a:wordpress:wordpress:5.7.2:*:*:*:*:*:*:*', 0.9805804786431419),
  ('cpe:2.3:a:wordpress:wordpress:-:*:*:*:*:*:*:*', 0.7071067811865475),
  ('cpe:2.3:a:adenion:blog2social:5.7.2:*:*:*:*:wordpress:*:*',
    0.6859944446591075)]
  ```
Finally, note that when *cpe_search* is used for the first time, it invokes a small setup routine that requests data from [NVD's official API](https://services.nvd.nist.gov/rest/json/cves/2.0) and precomputes the data utilized for searches in all subsequent runs. This may take a couple of minutes initially but is only done once.

## License
*cpe_search* is licensed under the MIT license, see [here](https://github.com/ra1nb0rn/cpe_search/blob/master/LICENSE).
