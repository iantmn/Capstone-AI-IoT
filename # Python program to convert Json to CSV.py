# Python program to convert
# JSON file to CSV


import json
import csv


# Opening JSON file and loading the data
# into the variable data
with open('output_path.json') as json_file:
	data = json.load(json_file)

Accelerometer_data = data
key = 'ACCL'

new_data = dict()
for key, value in Accelerometer_data.items():
    if not isinstance(value, dict):
        new_data[key] = value
    else:
        for k, v in value.items():
            new_data[key + "_" + k] = v

data_file = new_data

# now we will open a file for writing
data_file = open('data_file.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# Counter variable used for writing
# headers to the CSV file
count = 0

for emp in Accelerometer_data:
	if count == 0:

		# Writing headers of CSV file
		header = emp.keys()
		csv_writer.writerow(header)
		count += 1

	# Writing data of CSV file
	csv_writer.writerow(emp.values())

data_file.close()
