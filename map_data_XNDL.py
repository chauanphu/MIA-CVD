import os
import csv
import psycopg2
from datetime import datetime

# Database connection details
from config.db_configs import db_config

# Folder path containing the CSV files
folder_path = r"./dataset/xn"

# SQL to create the table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS medical_records_xn (
    id SERIAL PRIMARY KEY,
    IDXETNGHIEM VARCHAR(255),
    MAQL VARCHAR(255),
    MAVAOVIEN VARCHAR(255),
    PHAI INT,
    TUOI INT,
    ID_BV_TEN VARCHAR(255),
    TEN_BV_TEN TEXT,
    ID_BV_SO VARCHAR(255),
    TEN_BV_SO TEXT,
    DDMMYYYY DATE,
    ID_TEN VARCHAR(255),
    MA_TEN VARCHAR(255),
    TENXN TEXT,
    KETQUA VARCHAR(255),
    CANNANG FLOAT,
    CHIEUCAO FLOAT,
    HUYETAP VARCHAR(255),
    MACH FLOAT,
    NHIETDO FLOAT,
    MAXN VARCHAR(255),
    MAICD VARCHAR(255),
    CHANDOAN TEXT,
    LOAIBENHAN VARCHAR(255),
    MABN VARCHAR(255)
);
"""

# SQL to insert data into the medical_records_xn table
insert_query = """
INSERT INTO medical_records_xn (
    IDXETNGHIEM, MAQL, MAVAOVIEN, PHAI, TUOI, ID_BV_TEN, TEN_BV_TEN, ID_BV_SO, TEN_BV_SO, DDMMYYYY,
    ID_TEN, MA_TEN, TENXN, KETQUA, CANNANG, CHIEUCAO, HUYETAP, MACH, NHIETDO,
    MAXN, MAICD, CHANDOAN, LOAIBENHAN, MABN
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# Function to convert DDMMYYYY to a proper date format (YYYY-MM-DD)
def convert_to_date(date_value):
    try:
        date_string = str(date_value).strip()
        return datetime.strptime(date_string, '%d%m%Y').date()
    except (ValueError, TypeError):
        return None

# Function to clean row values (replace empty strings with None)
def clean_row(row):
    return [None if value == '' else value.strip() for value in row]

# Connect to the database
conn = None
cursor = None

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    # Delete the table if it exists
    # cursor.execute("DROP TABLE IF EXISTS medical_records_xn;")
    # Create the table if it does not exist
    cursor.execute(create_table_query)
    conn.commit()
    print("Table medical_records_xn created or already exists.")
    # Count how many files are in the folder
    file_count = len([name for name in os.listdir(folder_path) if name.endswith('.csv')])
    print(f"Number of CSV files in the folder: {file_count}")
    # Clear the table before importing new data
    # cursor.execute("TRUNCATE TABLE medical_records_xn;")
    # # Iterate through all CSV files in the folder
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".csv"):
    #         file_path = os.path.join(folder_path, filename)
    #         print(f'Processing file: {filename}')

    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             reader = csv.reader(file)
    #             headers = next(reader, None)  # Skip header row
    #             # Count the number of rows
    #             # row_count = sum(1 for _ in reader)
    #             # print(f"Number of rows in {filename}: {row_count}")
    #             for row in reader:
    #                 row = clean_row(row)
    #                 # Ensure row has exactly 24 columns
    #                 if len(row) == 24:
    #                     row[9] = convert_to_date(row[9])  # Convert DDMMYYYY

    #                     if row[9]:
    #                         cursor.execute(insert_query, row)
    #                     else:
    #                         print(f"Skipping row with invalid date: {row}")
    #                 else:
    #                     print(f"Skipping malformed row (Incorrect column count): {row}")

    #         conn.commit()
    #         print(f"Data from {filename} imported successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    if conn:
        conn.rollback()

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
