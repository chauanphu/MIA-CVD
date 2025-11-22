import os
import csv
import psycopg2
from datetime import datetime

# Database connection details
from config.db_configs import db_config

# Folder path containing the CDHA CSV files
folder_path = r"./dataset/cdha"

# Print connection details
print(f"Connecting to database {db_config['dbname']} at {db_config['host']}:{db_config['port']} as user {db_config['user']}")

# SQL to create the table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS medical_records_cdha (
    IDPHIEU VARCHAR(255) PRIMARY KEY,
    MAVAOVIEN VARCHAR(255),
    MAQL VARCHAR(255),
    PHAI INT,
    TUOI INT,
    MAICD VARCHAR(255),
    CHANDOAN TEXT,
    IDCDHA VARCHAR(255),
    KYTHUATCDHA TEXT,
    KETLUAN TEXT,
    MABN VARCHAR(255)
);
"""

# Clear data
truncate_table_query = "DELETE FROM medical_records_cdha;"

# SQL to insert data into the medical_records_cdha table
insert_query = """
INSERT INTO medical_records_cdha (
    IDPHIEU, MAVAOVIEN, MAQL, PHAI, TUOI, MAICD, CHANDOAN, IDCDHA, KYTHUATCDHA, KETLUAN, MABN
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (IDPHIEU) DO NOTHING;
"""

# Function to clean row values (replace empty strings with None)
def clean_row(row):
    return [None if value == '' else value.strip() for value in row]

# Connect to the database
# Connect to the database
conn = None
cursor = None


try:
    print("Connecting to the database...")
    conn = psycopg2.connect(**db_config)
    print("Connected to the database successfully.")
    cursor = conn.cursor()

    # Create the table if it does not exist
    cursor.execute(create_table_query)
    conn.commit()
    print("Table medical_records_cdha created or already exists.")
    # Clear existing data
    cursor.execute(truncate_table_query)
    # Iterate through all CSV files in the folder
    for filename in os.listdir(folder_path[:1]):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f'Processing file: {filename}')

            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                headers = next(reader, None)  # Skip header

                for row in reader:
                    row = clean_row(row)

                    # Ensure row has exactly 11 columns
                    if len(row) == 11:
                        cursor.execute(insert_query, row)
                    else:
                        print(f"Skipping malformed row (Incorrect column count): {row}")

            conn.commit()
            print(f"Data from {filename} imported successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    if conn:
        conn.rollback()

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()