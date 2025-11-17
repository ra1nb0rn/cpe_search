# cpe_search
Search for Common Platform Enumeration (CPE) strings using software names and titles.

## About
*cpe_search* can be used to search for Common Platform Enumeration (CPE) strings using software names and titles. For example, if some tool discovered a web server running *Apache 2.4.39*, you can use this tool to easily and quickly retrieve the corresponding CPE 2.3 string *cpe:2.3:<zero-width  space>a:apache:http_server:2.4.39:\*\:\*:\*:\*:\*:\*:\**. Thereafter, the retrieved CPE string can be used to accurately search for vulnerabilities, e.g. via the [Online NVD](https://nvd.nist.gov/) or the [search_vulns](https://github.com/ra1nb0rn/search_vulns) tool.

## Installation
You can install cpe_search via pip directly:
```
pip3 install cpe_search
```
You can also clone this repository and run:
```
pip3 install .
```

Note that when *cpe_search* is used for the first time, it invokes a small setup routine that downloads all available CPEs from the [NVD's official API](https://nvd.nist.gov/developers/products) and precomputes the data utilized for searches in all subsequent runs. This may take a couple of minutes initially but is only done once. To speed this process up, you can provide an NVD API key if you have one (it's free). The API key can be provided with the ``-k`` argument or specified in an environment variable called ``NVD_API_KEY``. You can also set up and provide a configuration file, see `config.json`.

## Usage
*cpe_search*'s usage information is shown in the following:
```
usage: cpe_search [-h] [-u] [-k API_KEY] [-n NUMBER] [-q QUERY] [-v] [-c CONFIG]

Search for CPEs using software names and titles -- Created by Dustin Born (ra1nb0rn)

options:
  -h, --help            show this help message and exit
  -u, --update          Update the local CPE database
  -k API_KEY, --api-key API_KEY
                        NVD API key to use for updating the local CPE dictionary
  -n NUMBER, --number NUMBER
                        The number of CPEs to show in the similarity overview (default: 3)
  -q QUERY, --query QUERY
                        A query, i.e. textual software name / title like 'Apache 2.4.39' or 'Wordpress 5.7.2'
  -v, --verbose         Be verbose and print status information
  -c CONFIG, --config CONFIG
                        A config file to use (default: config.json)
```
Note that when querying software with ``-q`` you have to put the software information in quotes if it contains any spaces. Also, you can use ``-q`` multiple times to make multiple queries at once. Moreover, the output can be piped to be directly useable with other tools. Here are some examples:
* Query *Sudo 1.8.2* to retrieve its CPE 2.3 string:
  ```bash
  $ cpe_search -q "Sudo 1.8.2"
  cpe:2.3:a:sudo_project:sudo:1.8.2:*:*:*:*:*:*:*
  [('cpe:2.3:a:sudo_project:sudo:1.8.2:*:*:*:*:*:*:*', 0.8660254037844385),
   ('cpe:2.3:a:sudo_project:sudo:1.3.0:*:*:*:*:*:*:*', 0.5773502691896256),
   ('cpe:2.3:a:cryptography.io:cryptography:1.8.2:*:*:*:*:*:*:*',
    0.4714045207910316)]
  ```
* Make a query and pipe the retrieved CPE to another tool:
  ```bash
  $ cpe_search -q "Windows 10 1809" | xargs echo
  cpe:2.3:o:microsoft:windows_10:1809:*:*:*:*:*:*:*
  ```
* Make two queries at once:
  ```bash
  $ cpe_search -q "Apache 2.4.39" -q "Wordpress 5.7.2"
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

## License
*cpe_search* is licensed under the MIT license, see [here](https://github.com/ra1nb0rn/cpe_search/blob/master/LICENSE).
