"""
ingestion.py
------------
Reads all five raw dataset files, logs metadata, and loads CSV tables
into PostgreSQL (fraud_db, schema: raw).

Run:
    python scripts/ingestion.py
"""

# Se importan todas las librerías necesarias para el script:
# os: manejo de rutas y archivos del sistema operativo
# sys: acceso a configuraciones del intérprete de Python (aquí se usa para ajustar la codificación)
# json: lectura y escritura de archivos JSON
# time: medición del tiempo de ejecución
# datetime: generación de marcas de tiempo (timestamps) para el log
# psycopg2 y psycopg2.extras: conector a PostgreSQL e inserción masiva de datos
# pandas: manipulación de datos tabulares
import os
import sys
import json
import time
import datetime
import psycopg2
import psycopg2.extras
import pandas as pd

# En Windows, la consola puede usar la codificación cp1252 por defecto,
# lo que causa errores al imprimir caracteres especiales (tildes, símbolos).
# Aquí se fuerza la salida estándar a usar UTF-8 si no lo hace ya.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Paths ────────────────────────────────────────────────────────────────────
# Se construye la ruta raíz del proyecto (dos niveles arriba del script actual)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Se apunta a la carpeta donde están almacenados los datos crudos
RAW_DIR   = os.path.join(BASE_DIR, "raw")
# Se define la ruta donde se guardará el log de ingesta en formato JSON
LOG_PATH  = os.path.join(BASE_DIR, "scripts", "ingestion_log.json")

# Se define un diccionario que mapea el nombre lógico de cada dataset a su ruta en disco.
# Hay tres archivos CSV (transacciones, tarjetas y usuarios) y dos JSON (códigos MCC y etiquetas de fraude).
FILES = {
    "transactions": os.path.join(RAW_DIR, "transactions_data.csv"),
    "cards":        os.path.join(RAW_DIR, "cards_data.csv"),
    "users":        os.path.join(RAW_DIR, "users_data.csv"),
    "mcc_codes":    os.path.join(RAW_DIR, "mcc_codes.json"),
    "fraud_labels": os.path.join(RAW_DIR, "train_fraud_labels.json"),
}

# ─── PostgreSQL connection params ─────────────────────────────────────────────
# Parámetros de conexión a la base de datos PostgreSQL que corre en Docker.
# Se usa el puerto 5433 para evitar conflictos con instalaciones locales de Postgres.
PG_PARAMS = dict(
    host="localhost",
    port=5433,
    user="postgres",
    password="postgres",
    database="fraud_db",
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read files and build ingestion log
# ──────────────────────────────────────────────────────────────────────────────

def read_files():
    """Read every dataset file, collect metadata, and return DataFrames."""
    print("\n" + "="*60)
    print("STEP 1 — Reading raw dataset files")
    print("="*60)

    # Se inicializan dos diccionarios: uno para el log de metadatos y otro para los DataFrames cargados
    log = {}
    dataframes = {}

    # Se itera sobre cada archivo definido en FILES para cargarlo y registrar sus metadatos
    for name, path in FILES.items():
        print(f"\n  → Loading '{name}' from {path}")
        # Se registra la marca de tiempo exacta en que se inicia la lectura del archivo
        ts = datetime.datetime.utcnow().isoformat()

        try:
            # Se calcula el tamaño del archivo en megabytes para incluirlo en el log
            size_mb = os.path.getsize(path) / (1024 * 1024)

            if path.endswith(".csv"):
                # Para archivos CSV se usa pandas con low_memory=False
                # para evitar advertencias por columnas con tipos mixtos
                df = pd.read_csv(path, low_memory=False)
                nrows = len(df)
                cols  = list(df.columns)
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Se normaliza el JSON a un DataFrame sin importar si viene como lista o diccionario.
                # Los archivos de etiquetas suelen ser {id: label, ...}, por eso se convierte a pares.
                if isinstance(raw, list):
                    df = pd.DataFrame(raw)
                elif isinstance(raw, dict):
                    # Some label files are {id: label, ...}
                    df = pd.DataFrame(list(raw.items()), columns=["id", "label"])
                else:
                    df = pd.DataFrame([raw])
                nrows = len(df)
                cols  = list(df.columns)
            else:
                raise ValueError(f"Unsupported file type: {path}")

            dataframes[name] = df

            # Se construye el registro de metadatos para este archivo en el log
            log[name] = {
                "file":        os.path.basename(path),
                "rows":        nrows,
                "columns":     cols,
                "size_mb":     round(size_mb, 4),
                "ingested_at": ts,
                "status":      "ok",
            }

            print(f"     Rows     : {nrows:,}")
            print(f"     Columns  : {cols}")
            print(f"     Size     : {size_mb:.4f} MB")
            print(f"     Timestamp: {ts}")

        except Exception as exc:
            # Si ocurre cualquier error al leer el archivo, se registra en el log como "error"
            # pero el script continúa con los demás archivos
            print(f"     ERROR reading {name}: {exc}")
            log[name] = {"file": os.path.basename(path), "status": "error", "error": str(exc)}

    return dataframes, log


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Save ingestion log
# ──────────────────────────────────────────────────────────────────────────────

# Esta función guarda el diccionario de metadatos como archivo JSON con formato legible (indent=2).
# El log sirve para auditar qué archivos se procesaron, cuándo y con cuántas filas.
def save_log(log: dict):
    print("\n" + "="*60)
    print("STEP 2 — Saving ingestion log")
    print("="*60)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓ Log saved to {LOG_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load CSVs into PostgreSQL raw schema
# ──────────────────────────────────────────────────────────────────────────────

# Esta función auxiliar convierte un tipo de dato de pandas al tipo equivalente en PostgreSQL.
# Se necesita para construir las sentencias CREATE TABLE de forma dinámica.
def _get_pg_type(dtype) -> str:
    """Map a pandas dtype to a simple PostgreSQL type."""
    s = str(dtype)
    if s.startswith("int"):
        return "BIGINT"
    if s.startswith("float"):
        return "DOUBLE PRECISION"
    if s.startswith("bool"):
        return "BOOLEAN"
    # Cualquier otro tipo (object, categoría, fechas no parseadas) se guarda como TEXT
    return "TEXT"


# Esta función carga los tres DataFrames CSV (transacciones, tarjetas, usuarios)
# en el esquema "raw" de PostgreSQL.
# Para cada tabla: limpia los nombres de columnas, genera el DDL dinámicamente
# y realiza la inserción en lotes de 5000 filas para mayor rendimiento.
def load_to_postgres(dataframes: dict):
    print("\n" + "="*60)
    print("STEP 3 — Loading raw tables into PostgreSQL (schema: raw)")
    print("="*60)

    try:
        conn = psycopg2.connect(**PG_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()
        print("  ✓ Connected to PostgreSQL")
    except Exception as exc:
        print(f"  ✗ Could not connect to PostgreSQL: {exc}")
        print("    (Make sure the docker-compose stack is running)")
        return

    # Se crea el esquema "raw" si no existe aún; aquí se guardan los datos sin transformar
    cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    print("  ✓ Schema 'raw' ensured")

    # Solo se cargan los tres CSVs principales; los JSON (mcc_codes y fraud_labels)
    # se manejan de forma separada en otros scripts
    table_map = {
        "transactions": "raw_transactions",
        "cards":        "raw_cards",
        "users":        "raw_users",
    }

    for key, table_name in table_map.items():
        df = dataframes.get(key)
        if df is None:
            print(f"  ✗ DataFrame '{key}' not available, skipping")
            continue

        # Se normalizan los nombres de columnas: minúsculas, sin espacios ni guiones
        df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        print(f"\n  → Loading {len(df):,} rows into raw.{table_name} ...")

        # Se construye la sentencia CREATE TABLE dinámicamente según los tipos del DataFrame.
        # Se hace DROP TABLE primero para reemplazar la tabla si ya existe de una ejecución anterior.
        col_defs = ",\n    ".join(
            f'"{col}" {_get_pg_type(dtype)}'
            for col, dtype in zip(df.columns, df.dtypes)
        )
        ddl = f'DROP TABLE IF EXISTS raw."{table_name}";\nCREATE TABLE raw."{table_name}" (\n    {col_defs}\n);'

        try:
            cur.execute(ddl)

            # Se convierte el DataFrame a una lista de tuplas para la inserción masiva.
            # execute_values es mucho más rápido que insertar fila por fila.
            rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
            cols_quoted = ", ".join(f'"{c}"' for c in df.columns)
            insert_sql  = f'INSERT INTO raw."{table_name}" ({cols_quoted}) VALUES %s'

            # Se insertan los datos en lotes de 5000 filas para controlar el uso de memoria
            BATCH = 5000
            for i in range(0, len(rows), BATCH):
                psycopg2.extras.execute_values(cur, insert_sql, rows[i:i+BATCH])

            print(f"     ✓ {len(df):,} rows inserted into raw.{table_name}")

        except Exception as exc:
            print(f"     ✗ Error loading {table_name}: {exc}")

    cur.close()
    conn.close()
    print("\n  ✓ PostgreSQL connection closed")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Punto de entrada del script. Se ejecutan los tres pasos en secuencia:
# 1. Lectura de archivos y construcción del log
# 2. Guardado del log en disco
# 3. Carga de las tablas en PostgreSQL
# Al final se imprime el tiempo total que tardó todo el proceso.
if __name__ == "__main__":
    start = time.time()
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          FRAUD DETECTION — DATA INGESTION            ║")
    print("╚══════════════════════════════════════════════════════╝")

    dataframes, log = read_files()
    save_log(log)
    load_to_postgres(dataframes)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Ingestion complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")
