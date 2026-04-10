from __future__ import annotations

import os
import sys
from pathlib import Path
os.getcwd()

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parsing.concat import concat_parsed, run_full_pipeline
from src.ingestion.chunker import process_chunking
from src.preprocessing.metadata_cleaning import process_metadata

import pandas as pd



#run_full_pipeline(folder_path="/home/shared/files", output_dir="/home/bidcoin/parsed")
#df_parsed = pd.read_csv("df_parsed.csv", encoding="utf-8")
#df = pd.read_csv("/home/bidcoin/parsed/data_list.csv", encoding="utf-8")
#df_meta = process_metadata(df, files_dir="/home/shared/files")
#df_meta.to_csv("data_list_metadata.csv", index=False, encoding="utf-8")
#process_chunking(df_parsed, df_meta)