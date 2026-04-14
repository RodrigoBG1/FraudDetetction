"""
create_schema.py
----------------
Creates the analytical star-schema tables in PostgreSQL (schema: curated)
and loads data from the enriched parquet files.

Run:
    python scripts/create_schema.py
"""

# Se importan las librerías necesarias:
# os: para manejar rutas de archivos y directorios del sistema
# json: para leer y escribir archivos en formato JSON
# psycopg2: conector que permite conectarse y ejecutar comandos en PostgreSQL
# psycopg2.extras: módulo auxiliar que ofrece funciones extra como inserción masiva de filas
# pandas: librería para manipular datos en forma de tablas (DataFrames)
import sys
import os
import json
import psycopg2
import psycopg2.extras
import pandas as pd

# Ensure Unicode output works on Windows terminals (cp1252 → utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Paths ────────────────────────────────────────────────────────────────────
# Se construye la ruta base del proyecto navegando dos niveles arriba desde este script
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Se define la carpeta donde se guardan los datos procesados
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
# Se apunta al directorio con el parquet enriquecido que generó el notebook de batch
ENRICHED_PATH = os.path.join(PROCESSED_DIR, "transactions_enriched")

# ─── PostgreSQL connection params ─────────────────────────────────────────────
# Se definen los parámetros de conexión a la base de datos PostgreSQL
# que corre localmente en el contenedor de Docker
PG_PARAMS = dict(
    host="localhost",
    port=5433,          # docker-compose maps host 5433 → container 5432
    user="postgres",
    password="postgres",
    database="fraud_db",
)

# ──────────────────────────────────────────────────────────────────────────────
# DDL for the curated star schema
# ──────────────────────────────────────────────────────────────────────────────

# Este bloque de texto contiene el SQL que define el esquema estrella en PostgreSQL.
# Un esquema estrella tiene tablas de dimensiones (dim_*) que describen entidades
# y una tabla de hechos (fact_transactions) que registra los eventos (transacciones).
# Se usa CASCADE al hacer DROP para eliminar también las claves foráneas que dependen.
DDL = """
CREATE SCHEMA IF NOT EXISTS curated;

-- Dimension: users
DROP TABLE IF EXISTS curated.dim_users CASCADE;
CREATE TABLE curated.dim_users (
    user_id      TEXT PRIMARY KEY,
    age          INTEGER,
    location     TEXT,
    account_type TEXT
);

-- Dimension: cards
DROP TABLE IF EXISTS curated.dim_cards CASCADE;
CREATE TABLE curated.dim_cards (
    card_id         TEXT PRIMARY KEY,
    card_type       TEXT,
    card_limit      DOUBLE PRECISION,
    activation_date TEXT
);

-- Dimension: merchants (MCC)
DROP TABLE IF EXISTS curated.dim_merchants CASCADE;
CREATE TABLE curated.dim_merchants (
    mcc_code             TEXT PRIMARY KEY,
    merchant_description TEXT,
    mcc_fraud_rate       DOUBLE PRECISION
);

-- Fact: transactions
DROP TABLE IF EXISTS curated.fact_transactions CASCADE;
CREATE TABLE curated.fact_transactions (
    transaction_id       TEXT PRIMARY KEY,
    user_id              TEXT REFERENCES curated.dim_users(user_id),
    card_id              TEXT REFERENCES curated.dim_cards(card_id),
    mcc_code             TEXT REFERENCES curated.dim_merchants(mcc_code),
    amount               DOUBLE PRECISION,
    timestamp            TIMESTAMP,
    is_weekend           BOOLEAN,
    is_night             BOOLEAN,
    amount_deviation     DOUBLE PRECISION,
    transaction_velocity DOUBLE PRECISION,
    card_utilization     DOUBLE PRECISION,
    is_fraud             INTEGER
);
"""


# Esta función establece la conexión con PostgreSQL usando los parámetros definidos arriba.
# Si la conexión falla, se lanza un error con el mensaje descriptivo.
# autocommit=True hace que cada instrucción SQL se confirme automáticamente sin necesidad de commit manual.
def connect():
    try:
        conn = psycopg2.connect(**PG_PARAMS)
        conn.autocommit = True
        print("  ✓ Connected to PostgreSQL fraud_db")
        return conn
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to PostgreSQL: {exc}")


# Esta función ejecuta el DDL completo para crear el esquema curated y todas sus tablas.
# Se abre un cursor (objeto que permite enviar comandos SQL) y se ejecuta el bloque DDL.
def create_schema(conn):
    print("\n" + "="*60)
    print("STEP 1 — Creating curated schema and tables")
    print("="*60)
    cur = conn.cursor()
    cur.execute(DDL)
    print("  ✓ Schema 'curated' and all tables created")
    cur.close()


# Esta función lee el archivo parquet enriquecido que fue generado en el notebook de batch processing.
# Primero verifica que el archivo exista; si no, lanza un error informativo.
# Luego normaliza los nombres de columnas a minúsculas y sin espacios para evitar inconsistencias.
def load_enriched() -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 2 — Loading enriched parquet")
    print("="*60)

    if not os.path.exists(ENRICHED_PATH):
        raise FileNotFoundError(
            f"Enriched parquet not found at {ENRICHED_PATH}.\n"
            "Please run notebooks/batch_processing.ipynb first."
        )

    df = pd.read_parquet(ENRICHED_PATH)
    # Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    print(f"  ✓ Loaded {len(df):,} rows from {ENRICHED_PATH}")
    return df


# Esta función realiza la inserción masiva de filas en una tabla de PostgreSQL.
# Usa execute_values para enviar múltiples filas en un solo comando SQL,
# lo que es mucho más eficiente que insertar fila por fila.
# ON CONFLICT DO NOTHING evita errores si alguna fila ya existe en la tabla.
def insert_batch(cur, table: str, rows: list, cols: list):
    if not rows:
        return
    cols_q   = ", ".join(f'"{c}"' for c in cols)
    insert   = f'INSERT INTO {table} ({cols_q}) VALUES %s ON CONFLICT DO NOTHING'
    psycopg2.extras.execute_values(cur, insert, rows, page_size=2000)


# Esta función carga las tres tablas de dimensiones del esquema estrella:
# dim_users, dim_cards y dim_merchants.
# Para cada una, se seleccionan las columnas relevantes del DataFrame,
# se eliminan duplicados y se rellenan con None las columnas que no existan en los datos.
def load_dimensions(conn, df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 3 — Loading dimension tables")
    print("="*60)
    cur = conn.cursor()

    # ── dim_users ──────────────────────────────────────────────────────────
    # Resolve actual column names from the parquet (may differ from schema names)
    def first_col(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    uid_src = first_col("user_id", "client_id", "customer_id")
    age_src = first_col("age", "current_age")
    loc_src = first_col("location", "address")
    acc_src = first_col("account_type")

    user_cols = ["user_id", "age", "location", "account_type"]
    if uid_src is None:
        print("  ⚠ dim_users: no user_id column found — skipping")
    else:
        user_rename = {k: v for k, v in {
            uid_src: "user_id", age_src: "age", loc_src: "location", acc_src: "account_type",
        }.items() if k is not None}
        users_df = df[[c for c in [uid_src, age_src, loc_src, acc_src] if c]].copy()
        users_df = users_df.rename(columns=user_rename)
        users_df = users_df.drop_duplicates(subset=["user_id"]).dropna(subset=["user_id"])
        for c in user_cols:
            if c not in users_df.columns:
                users_df[c] = None
        users_df = users_df[user_cols]
        rows = [tuple(r) for r in users_df.itertuples(index=False, name=None)]
        insert_batch(cur, "curated.dim_users", rows, user_cols)
        print(f"  ✓ dim_users: {len(users_df):,} rows inserted")

    # ── dim_cards ──────────────────────────────────────────────────────────
    cid_src   = first_col("card_id")
    ctype_src = first_col("card_type", "card_brand")
    clim_src  = first_col("card_limit", "credit_limit")
    cact_src  = first_col("activation_date", "acct_open_date")

    card_cols  = ["card_id", "card_type", "card_limit", "activation_date"]
    avail_card = [c for c in [cid_src, ctype_src, clim_src, cact_src] if c]
    card_rename = {k: v for k, v in {
        cid_src: "card_id", ctype_src: "card_type", clim_src: "card_limit", cact_src: "activation_date",
    }.items() if k is not None}
    cards_df   = df[avail_card].copy().rename(columns=card_rename)
    cards_df   = cards_df.drop_duplicates(subset=["card_id"]).dropna(subset=["card_id"])
    # Strip currency symbols/commas and coerce to float for the DOUBLE PRECISION column
    if "card_limit" in cards_df.columns:
        cards_df["card_limit"] = (
            cards_df["card_limit"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
    for c in card_cols:
        if c not in cards_df.columns:
            cards_df[c] = None
    cards_df = cards_df[card_cols]
    rows = [tuple(r) for r in cards_df.itertuples(index=False, name=None)]
    insert_batch(cur, "curated.dim_cards", rows, card_cols)
    print(f"  ✓ dim_cards: {len(cards_df):,} rows inserted")

    # ── dim_merchants ──────────────────────────────────────────────────────
    # Se busca dinámicamente la columna de código MCC en el DataFrame,
    # ya que su nombre puede variar entre distintas versiones del dataset.
    # Se excluyen columnas que contengan "fraud" o "rate" para no confundirlas con el MCC puro.
    mcc_col = next((c for c in df.columns if "mcc" in c.lower() and "fraud" not in c.lower() and "rate" not in c.lower()), None)
    desc_col = next((c for c in df.columns if "description" in c.lower() or "merchant_desc" in c.lower()), None)

    if mcc_col:
        mcc_cols = [mcc_col]
        if desc_col:
            mcc_cols.append(desc_col)
        if "mcc_fraud_rate" in df.columns:
            mcc_cols.append("mcc_fraud_rate")

        # Se renombran las columnas al formato estándar esperado por la tabla dim_merchants
        mcc_df = df[mcc_cols].drop_duplicates(subset=[mcc_col]).dropna(subset=[mcc_col]).rename(
            columns={mcc_col: "mcc_code", desc_col: "merchant_description"} if desc_col else {mcc_col: "mcc_code"}
        )
        for c in ["mcc_code", "merchant_description", "mcc_fraud_rate"]:
            if c not in mcc_df.columns:
                mcc_df[c] = None
        mcc_df = mcc_df[["mcc_code", "merchant_description", "mcc_fraud_rate"]]
        # Se convierte mcc_code a string para que coincida con el tipo TEXT de la tabla
        mcc_df["mcc_code"] = mcc_df["mcc_code"].astype(str)
        rows = [tuple(r) for r in mcc_df.itertuples(index=False, name=None)]
        insert_batch(cur, "curated.dim_merchants", rows, ["mcc_code", "merchant_description", "mcc_fraud_rate"])
        print(f"  ✓ dim_merchants: {len(mcc_df):,} rows inserted")
    else:
        print("  ! dim_merchants: no MCC column found, skipping")

    cur.close()


# Esta función carga la tabla de hechos fact_transactions.
# Primero detecta de forma flexible el nombre de cada columna relevante,
# ya que los nombres pueden variar entre versiones del dataset.
# Luego construye un DataFrame con las columnas exactas que espera la tabla,
# rellenando con None las que no existan.
def load_facts(conn, df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 4 — Loading fact_transactions")
    print("="*60)
    cur = conn.cursor()

    # Esta función auxiliar busca el primer nombre de columna disponible
    # entre una lista de candidatos posibles
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Se detecta el nombre real de cada columna usando nombres alternativos comunes
    txn_id_col  = find_col(["transaction_id", "id", "trans_id", "index"])
    user_id_col = find_col(["user_id", "client_id", "customer_id"])
    card_id_col = find_col(["card_id"])
    mcc_col     = find_col([c for c in df.columns if "mcc" in c.lower() and "fraud" not in c.lower() and "rate" not in c.lower()])
    amount_col  = find_col(["amount"])
    ts_col      = find_col(["timestamp", "date", "trans_date", "transaction_date"])

    # Se mapea cada columna destino (en la tabla SQL) a su columna fuente (en el DataFrame)
    fact_cols_map = {
        "transaction_id":       txn_id_col,
        "user_id":              user_id_col,
        "card_id":              card_id_col,
        "mcc_code":             mcc_col,
        "amount":               amount_col,
        "timestamp":            ts_col,
        "is_weekend":           find_col(["is_weekend"]),
        "is_night":             find_col(["is_night"]),
        "amount_deviation":     find_col(["amount_deviation"]),
        "transaction_velocity": find_col(["transaction_velocity"]),
        "card_utilization":     find_col(["card_utilization"]),
        "is_fraud":             find_col(["is_fraud", "fraud", "label"]),
    }

    # Se construye el DataFrame de hechos copiando cada columna o asignando None si no existe
    fact_df = pd.DataFrame()
    for target, source in fact_cols_map.items():
        if source and source in df.columns:
            fact_df[target] = df[source]
        else:
            fact_df[target] = None

    # Cast mcc_code to string to match dimension FK
    if "mcc_code" in fact_df.columns:
        fact_df["mcc_code"] = fact_df["mcc_code"].astype(str).where(fact_df["mcc_code"].notna(), None)

    # Si no existe una columna de ID de transacción, se generan IDs numéricos como strings
    if txn_id_col is None or fact_df["transaction_id"].isna().all():
        fact_df["transaction_id"] = [str(i) for i in range(len(fact_df))]
    else:
        fact_df["transaction_id"] = fact_df["transaction_id"].astype(str)

    # Drop rows where the primary key is null — PostgreSQL rejects them
    fact_df = fact_df.dropna(subset=["transaction_id"])

    # Se convierten los NaN de pandas a None de Python antes de insertar,
    # porque psycopg2 no entiende float NaN como valor nulo de SQL
    rows = []
    for r in fact_df.itertuples(index=False, name=None):
        rows.append(tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in r))

    # Se insertan los datos en lotes de 5000 filas para no saturar la memoria
    BATCH = 5000
    for i in range(0, len(rows), BATCH):
        insert_batch(cur, "curated.fact_transactions", rows[i:i+BATCH], list(fact_df.columns))

    print(f"  ✓ fact_transactions: {len(fact_df):,} rows inserted")
    cur.close()


# Esta función consulta el conteo de filas en cada tabla del esquema curated
# para verificar que la carga se realizó correctamente.
def print_counts(conn):
    print("\n" + "="*60)
    print("STEP 5 — Row counts confirmation")
    print("="*60)
    cur = conn.cursor()
    tables = ["curated.dim_users", "curated.dim_cards", "curated.dim_merchants", "curated.fact_transactions"]
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        count = cur.fetchone()[0]
        print(f"  {t:<40} {count:>10,} rows")
    cur.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Punto de entrada principal del script.
# Se ejecutan en orden todos los pasos: conexión, creación del esquema,
# carga del parquet, inserción de dimensiones, inserción de hechos y verificación final.
# Se mide el tiempo total de ejecución con time.time().
if __name__ == "__main__":
    import time
    start = time.time()
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        FRAUD DETECTION — CREATE CURATED SCHEMA       ║")
    print("╚══════════════════════════════════════════════════════╝")

    try:
        conn = connect()
        create_schema(conn)
        df   = load_enriched()
        load_dimensions(conn, df)
        load_facts(conn, df)
        print_counts(conn)
        conn.close()
    except Exception as exc:
        print(f"\n  ✗ Fatal error: {exc}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Schema creation complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")
