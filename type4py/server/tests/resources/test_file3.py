"""
This file is given by Ali Khatami (@akhatami) who found a bug!
"""

import json
from pprint import pprint

def find_plugin(plugin_name: str, content_list: List[str]):
    found = False
    for index, line in enumerate(contents):
        if 'plugins {' in line:
            jacoco_index: int = index + 1

f = open('../sample_build_gradle/JDA/build.gradle.kts', 'r')
contents = f.readlines()
jacoco_index = -1
for index, line in enumerate(contents):
    if 'plugins {' in line:
        jacoco_index = index + 1
    
contents.insert(jacoco_index, '    jacoco')
pprint(contents)
f.close()