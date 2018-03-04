import json
import requests
import time
import random

HERO_ENDPOINT = 'https://api.opendota.com/api/heroes?'
REG_MATCH_BATCH_ENDPOINT = 'https://api.opendota.com/api/publicMatches'

REG_MATCH_FILE = 'data/reg_matches.json'
HERO_PATH = 'data/heroes.json'

#### Functions #################################################################

def get_heroes():
    with open(HERO_PATH, 'w') as f:
        f.write(requests.get(HERO_ENDPOINT).text)


def load_heroes():
        with open(HERO_PATH, 'r') as f:
            return json.loads(f.read().strip())


def previous_progress_matches(infile):
    match_ids = []
    with open(infile, 'r') as f:
        for line in f:
            if line == 'error\n':
                continue
            try:
                j = json.loads(line)
                match_ids.append(int(j['match_id']))
            except:
                print(j)
    if len(match_ids) > 0:
        return max(match_ids), min(match_ids)
    else:
        return None, None


def query_matches(endpoint, less_than_match_id):
    if less_than_match_id == None:
        request_string = endpoint
    else:
        request_string = endpoint +\
            '?less_than_match_id={}'.format(less_than_match_id)
    print(request_string)
    return json.loads(requests.get(request_string).text)


def get_least_recent_match_id(list_of_matches):
    match_ids = []
    for match in list_of_matches:
        if match == 'error':
            continue
        match_ids.append(match['match_id'])
    try:
        return min(match_ids)
    except:
        return []


def write_matches(outfile, matches_list):
    with open(outfile, 'a') as f:
        for line in matches_list:
            f.write(json.dumps(line) + '\n')


#### Get heroes ################################################################

get_heroes()

#### Get matches ###############################################################

most_recent_match, most_distant_match = previous_progress_matches(
    REG_MATCH_FILE
)

while True:
    try:
        results = query_matches(
            REG_MATCH_BATCH_ENDPOINT,
            most_distant_match,
        )
        time.sleep(1.5)
        write_matches(REG_MATCH_FILE, results)
        most_distant_match = get_least_recent_match_id(results)
    except:
        time.sleep(10)
        continue
    print('Queried!')
