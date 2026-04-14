"""
load_mongo.py
-------------
Loads MCC codes and merchant profile aggregations into MongoDB (fraud_mongo).

Collections created:
  - mcc_lookup        : raw records from mcc_codes.json
  - merchant_profiles : aggregated stats per MCC code

Run:
    python scripts/load_mongo.py
"""

# Se importan las librerías necesarias:
# os: manejo de rutas del sistema de archivos
# json: lectura del archivo mcc_codes.json
# time: medición del tiempo total de ejecución
# pandas: procesamiento del parquet enriquecido para calcular estadísticas por MCC
# pymongo: cliente oficial de Python para conectarse a MongoDB e insertar documentos
# pymongo.errors.ConnectionFailure: excepción específica que se lanza si MongoDB no está disponible
import os
import sys
import json
import time
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

# Ensure stdout uses UTF-8 so Unicode box-drawing characters render on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Paths ────────────────────────────────────────────────────────────────────
# Se define la ruta raíz del proyecto navegando dos niveles arriba desde este script
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
# Ruta al archivo JSON con los códigos MCC (categorías de comercios)
MCC_JSON      = os.path.join(RAW_DIR, "mcc_codes.json")
# Ruta al parquet enriquecido generado en el notebook de batch processing
ENRICHED_PATH = os.path.join(PROCESSED_DIR, "transactions_enriched")


# ─── MongoDB connection ───────────────────────────────────────────────────────
# Esta función establece la conexión con MongoDB usando el cliente de pymongo.
# Se usa serverSelectionTimeoutMS=5000 para que falle rápido si MongoDB no está corriendo,
# en lugar de esperar indefinidamente.
# El comando "ping" verifica que la conexión sea real y no solo aparente.
def connect_mongo() -> MongoClient:
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print("  ✓ Connected to MongoDB at localhost:27017")
        return client
    except ConnectionFailure as exc:
        raise RuntimeError(f"Cannot connect to MongoDB: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load mcc_codes.json → mcc_lookup collection
# ──────────────────────────────────────────────────────────────────────────────

# Esta función carga el archivo mcc_codes.json en la colección "mcc_lookup" de MongoDB.
# El archivo puede estar en dos formatos distintos:
#   - Lista de diccionarios: se usa directamente
#   - Diccionario {código: descripción}: se convierte a lista de documentos
# Se normaliza el nombre del campo a "mcc_code" para tener consistencia.
# La colección se elimina y recrea en cada ejecución para garantizar datos frescos.
# Se crea un índice único sobre mcc_code para acelerar búsquedas y evitar duplicados.
def load_mcc_lookup(db):
    print("\n" + "="*60)
    print("STEP 1 — Loading mcc_codes.json into mcc_lookup collection")
    print("="*60)

    with open(MCC_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Se normaliza el formato de entrada a una lista de diccionarios
    if isinstance(raw, list):
        docs = raw
    elif isinstance(raw, dict):
        docs = [{"mcc_code": k, "description": v} for k, v in raw.items()]
    else:
        docs = [raw]

    # Se recorren todos los documentos para asegurarse de que el campo mcc_code
    # esté presente y sea un string, sin importar cómo se llame el campo original
    for d in docs:
        for key in list(d.keys()):
            if "mcc" in key.lower() or "code" in key.lower():
                d["mcc_code"] = str(d[key])
                break

    # Se elimina la colección anterior y se recrea con un índice único sobre mcc_code
    collection = db["mcc_lookup"]
    collection.drop()
    collection.create_index([("mcc_code", ASCENDING)], unique=True)
    collection.insert_many(docs)

    count = collection.count_documents({})
    print(f"  ✓ mcc_lookup: {count:,} documents inserted")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build merchant_profiles from enriched parquet
# ──────────────────────────────────────────────────────────────────────────────

# Esta función construye la colección "merchant_profiles" con estadísticas agregadas por código MCC.
# Si el parquet enriquecido existe, se calculan métricas reales como:
#   - total de transacciones por categoría de comercio
#   - cantidad y tasa de fraude
#   - monto promedio de transacción
# Si no existe el parquet, se generan perfiles vacíos usando solo los datos del mcc_lookup.
def load_merchant_profiles(db, mcc_docs: list):
    print("\n" + "="*60)
    print("STEP 2 — Building merchant_profiles collection")
    print("="*60)

    # Se construye un diccionario de búsqueda {mcc_code: descripción} a partir de los documentos ya cargados
    desc_map = {}
    for d in mcc_docs:
        code = str(d.get("mcc_code", ""))
        # Se intenta obtener la descripción de varios campos posibles
        desc = d.get("description", d.get("edited_description", ""))
        if code:
            desc_map[code] = desc

    if not os.path.exists(ENRICHED_PATH):
        # Si no se encuentra el parquet, se generan perfiles con todas las métricas en cero
        print(f"  ! Enriched parquet not found at {ENRICHED_PATH}")
        print("    Falling back to empty profiles from mcc_lookup data only")
        profiles = []
        for d in mcc_docs:
            code = str(d.get("mcc_code", ""))
            profiles.append({
                "mcc_code":            code,
                "description":         desc_map.get(code, ""),
                "total_transactions":  0,
                "fraud_count":         0,
                "fraud_rate":          0.0,
                "avg_transaction_amount": 0.0,
            })
    else:
        # Se lee el parquet enriquecido y se normalizan los nombres de columnas
        df = pd.read_parquet(ENRICHED_PATH)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Se detectan dinámicamente los nombres de columna relevantes
        mcc_col    = next((c for c in df.columns if "mcc" in c and "fraud" not in c and "rate" not in c), None)
        amount_col = next((c for c in df.columns if c == "amount"), None)
        fraud_col  = next((c for c in df.columns if c in ("is_fraud", "fraud", "label")), None)

        if mcc_col is None:
            print("  ! No MCC column found in enriched data; creating empty profiles")
            profiles = []
        else:
            # Se convierte el código MCC a string para uniformidad con el mcc_lookup
            df[mcc_col] = df[mcc_col].astype(str)

            # Se agrupa por código MCC para calcular las métricas de cada categoría de comercio
            grp = df.groupby(mcc_col)
            total = grp.size().rename("total_transactions")

            agg = pd.DataFrame({"total_transactions": total})

            if fraud_col:
                # Se calcula la cantidad de transacciones fraudulentas y la tasa de fraude por MCC
                fraud_cnt = grp[fraud_col].sum().rename("fraud_count")
                agg = agg.join(fraud_cnt)
                agg["fraud_rate"] = agg["fraud_count"] / agg["total_transactions"]
            else:
                agg["fraud_count"] = 0
                agg["fraud_rate"]  = 0.0

            if amount_col:
                # Se calcula el monto promedio de transacción por categoría de comercio
                avg_amt = grp[amount_col].mean().rename("avg_transaction_amount")
                agg = agg.join(avg_amt)
            else:
                agg["avg_transaction_amount"] = 0.0

            agg = agg.reset_index().rename(columns={mcc_col: "mcc_code"})

            # Se construye la lista de documentos que se insertarán en MongoDB,
            # asegurando que los tipos sean nativos de Python (int, float, str)
            # porque MongoDB no acepta tipos de numpy directamente
            profiles = []
            for _, row in agg.iterrows():
                code = str(row["mcc_code"])
                profiles.append({
                    "mcc_code":               code,
                    "description":            desc_map.get(code, ""),
                    "total_transactions":     int(row["total_transactions"]),
                    "fraud_count":            int(row.get("fraud_count", 0)),
                    "fraud_rate":             float(row.get("fraud_rate", 0.0)),
                    "avg_transaction_amount": float(row.get("avg_transaction_amount", 0.0)),
                })

    # Se elimina la colección anterior y se crea con índice único antes de insertar
    collection = db["merchant_profiles"]
    collection.drop()
    collection.create_index([("mcc_code", ASCENDING)], unique=True)

    if profiles:
        collection.insert_many(profiles)

    count = collection.count_documents({})
    print(f"  ✓ merchant_profiles: {count:,} documents inserted")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Punto de entrada del script. Se conecta a MongoDB, se selecciona la base de datos "fraud_mongo"
# y se ejecutan los dos pasos en secuencia:
# 1. Carga del catálogo de códigos MCC en la colección mcc_lookup
# 2. Construcción de perfiles de comerciantes con estadísticas de fraude en merchant_profiles
# Al final se imprimen los conteos de documentos en cada colección como verificación.
if __name__ == "__main__":
    start = time.time()
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         FRAUD DETECTION — LOAD MONGODB               ║")
    print("╚══════════════════════════════════════════════════════╝")

    try:
        client = connect_mongo()
        # Se selecciona (o crea si no existe) la base de datos de fraude en MongoDB
        db     = client["fraud_mongo"]

        mcc_docs = load_mcc_lookup(db)
        load_merchant_profiles(db, mcc_docs)

        print("\n" + "="*60)
        print("  Final collection counts:")
        for col in ["mcc_lookup", "merchant_profiles"]:
            print(f"    {col:<30} {db[col].count_documents({}):>8,} documents")

        client.close()
    except Exception as exc:
        print(f"\n  ✗ Fatal error: {exc}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  MongoDB load complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")
