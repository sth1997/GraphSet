import re
import csv

def read_time_cost(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        pattern = r"Counting time cost: (\d+\.\d+) s"
        match = re.search(pattern, content)

        if match:
            time_cost = float(match.group(1))
            return time_cost
        else:
            print(f"Time not found in file {file_path}.")
            return None

def write_table(table, csv_file):
    # Write table and header into a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)

