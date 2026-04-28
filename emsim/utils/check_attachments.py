import re
import requests

urls = [
    "https://emsimcases.com/2015/12/08/hyponatremic-seizure/hyponatremic-seizure/",
    "https://emsimcases.com/2015/11/24/dka/dka-case-2/",
    "https://emsimcases.com/2016/01/05/two-patient-trauma/two-for-one-mvc/",
]
for url in urls:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    links = re.findall(r'href=[\'"]([^\'"]+\.(?:docx|doc|pdf))[\'"]', r.text, re.I)
    print(url)
    for l in set(links):
        print("  ", l)
