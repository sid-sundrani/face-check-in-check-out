from dateutil.parser import *
import json
from datetime import datetime

check_in = json.load(open('check_in_times.json'))
check_out = json.load(open('check_out_times.json'))

names_in = list(check_in.keys())
names_out = list(check_out.keys())
check_details = {}


for name, times_in in check_in.items():
    times_out = check_out[name]
    check_details[name] = {'first in': times_in[0], 'last_out': times_out[-1]}


print(check_details)