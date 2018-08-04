from dateutil.parser import *
import json
from datetime import datetime

data = json.load(open('check_in_times.json'))
date = data['Sid']['check-in']
a = parse(date[0])
b = datetime.now()
print(b-a)
