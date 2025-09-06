#!/usr/bin/env python3
"""既存の専門用語を全削除するスクリプト"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.config import Config

load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"

engine = create_engine(PG_URL)

with engine.begin() as conn:
    result = conn.execute(text("DELETE FROM jargon_dictionary"))
    print(f"Deleted {result.rowcount} terms from jargon_dictionary table")