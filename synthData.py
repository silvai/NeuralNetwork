import sys
import csv
from numpy import random


size = int(sys.argv[1])
exp = int(sys.argv[2])
input_data = []
output_data = []
filename = "testData.csv"

for i in range(size):
    input_data.append(random.randint(0, 1000))
for d in input_data:
    output_data.append((d**exp)*random.randint(0, 1000))

with open(filename, 'wt') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(input_data)
    writer.writerow(output_data)
    print(input_data)
    print(output_data)

