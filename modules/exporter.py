import pandas as pd
import json
import yaml
import sqlite3
import tempfile
import os
from io import BytesIO, StringIO


#  CSV 
def to_csv(df):
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue()


#  EXCEL 
def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")

    buffer.seek(0)
    return buffer.getvalue()  # FIXED


#  JSON 
def to_json(df):
    return df.to_json(orient="records", indent=4, force_ascii=False)


#  XML 
def to_xml(df):
    import re
    df_xml = df.copy()
    df_xml.columns = [re.sub(r'\W', '_', str(c)) if not str(c)[:1].isdigit() else '_' + re.sub(r'\W', '_', str(c)) for c in df.columns]
    return df_xml.to_xml(index=False)


#  YAML 
def to_yaml(df):
    data = df.to_dict(orient="records")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


#  SQL 
def to_sqlite(df, table_name="data"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")

    try:
        conn = sqlite3.connect(temp_file.name)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()

        with open(temp_file.name, "rb") as f:
            data = f.read()

    finally:
        os.remove(temp_file.name)  # FIXED (no leak)

    return data


#  UNIVERSAL EXPORT 
def export_data(df, file_format):
    """
    file_format:
    - csv
    - excel
    - json
    - xml
    - yaml
    - sql
    """

    if df is None or df.empty:
        return None, "Empty DataFrame"

    df = df.copy()
    df.columns = df.columns.str.strip()

    file_format = file_format.lower().strip()

    supported_formats = ["csv", "excel", "json", "xml", "yaml", "sql"]

    if file_format not in supported_formats:
        return None, f"Unsupported format. Choose from {supported_formats}"

    try:
        if file_format == "csv":
            return to_csv(df), "text/csv"

        elif file_format == "excel":
            return to_excel(df), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif file_format == "json":
            return to_json(df), "application/json"

        elif file_format == "xml":
            return to_xml(df), "application/xml"

        elif file_format == "yaml":
            return to_yaml(df), "text/yaml"

        elif file_format == "sql":
            return to_sqlite(df), "application/octet-stream"

    except Exception as e:
        return None, f"Error: {str(e)}"