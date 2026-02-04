import os
import sqlite3

import pandas as pd

from graph.consts import DB_PATH


def inspect_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        return

    print(f"Inspecting Database: {DB_PATH}\n" + "=" * 50)

    # Connect to the database
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()

    # 1. Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    if not tables:
        print("No tables found in the database.")
        con.close()
        return

    print(f"Found {len(tables)} tables: {', '.join(tables)}\n")

    # 2. Inspect each table
    for table in tables:
        print(f"TABLE: {table}")
        print("-" * 30)

        # Get Column Info (PRAGMA table_info gives cid, name, type, notnull, dflt_value, pk)
        columns_info = cursor.execute(f"PRAGMA table_info({table})").fetchall()

        # Get Row Count
        row_count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # Display Columns
        print(f"   Rows: {row_count:,}")
        print("   Columns:")
        for col in columns_info:
            cid, name, dtype, notnull, _, _ = col
            req_str = "NOT NULL" if notnull else "NULLABLE"
            print(f"     - {name} ({dtype})")

        # Optional: Preview Data using Pandas for nice formatting
        print("\n   Sample Data (First 3 rows):")
        try:
            df_sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3", con)
            print(df_sample.to_string(index=False))
        except Exception as e:
            print(f"     [Could not load sample data: {e}]")

        print("\n" + "=" * 50 + "\n")

    con.close()
    print("Inspection Complete.")


if __name__ == "__main__":
    inspect_database()
