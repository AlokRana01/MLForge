import pandas as pd
import json
import yaml
import sqlite3
import xml.etree.ElementTree as ET
import tempfile
import os


def load_file(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        # Reset pointer
        uploaded_file.seek(0)

        #  CSV 
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")

        #  EXCEL 
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        #  JSON 
        elif filename.endswith(".json"):
            data = json.load(uploaded_file)

            # Flatten nested JSON
            df = pd.json_normalize(data)

        #  XML 
        elif filename.endswith(".xml"):
            tree = ET.parse(uploaded_file)
            root = tree.getroot()

            data = []
            for child in root:
                row = {}

                # child elements
                for sub in child:
                    row[sub.tag] = sub.text

                # attributes
                for key, val in child.attrib.items():
                    row[key] = val

                data.append(row)

            df = pd.DataFrame(data)

        #  YAML 
        elif filename.endswith((".yaml", ".yml")):
            data = yaml.safe_load(uploaded_file)
            df = pd.json_normalize(data)

        #  SQLITE 
        elif filename.endswith(".db"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            conn = sqlite3.connect(tmp_path)

            # Get first table name
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", conn
            )

            if tables.empty:
                return None, "No tables found in database"

            table_name = tables.iloc[0, 0]

            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

            conn.close()
            os.remove(tmp_path)

        else:
            return None, "Unsupported file format"

        # Final check
        if df.empty:
            return None, "Loaded file is empty"

        return df, "Success"

    except Exception as e:
        return None, f"Error: {str(e)}"