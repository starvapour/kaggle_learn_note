import pandas as pd
from collections import Counter

csv_data = pd.read_csv("train.csv")
label_count = Counter(list(csv_data['label']))
print(label_count)

'''
Counter({3: 13158, 4: 2577, 2: 2386, 1: 2189, 0: 1087})
'''