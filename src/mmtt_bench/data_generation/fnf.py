import argparse
import glob
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

EMPTY_STRING_FILL = "Sorry, no available information"

T_COL   = 'Date'
Y_COL   = 'OT'
TEXT_COL = 'Final_Search_2'


def clean_search_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^Available facts are as follows:\s*', '', text)
    text = text.rstrip(';').strip()
    facts = re.split(r'\d{4}-\d{2}-\d{2}:\s*', text)
    facts = [f.strip().rstrip('.').strip() for f in facts
             if f.strip() and f.strip() not in ('NA', 'NA.')]
    return ' '.join(facts)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic):  return obj.item()
        return super().default(obj)


# def _sample_irrelevant_texts(n: int, data_name: str, text_dict: dict,
#                               rng: np.random.Generator) -> list:
#     other_keys = [p for p in text_dict.keys() if p != data_name]
#     other_lists = [text_dict[k] for k in other_keys]
#     path_idxs = rng.integers(len(other_keys), size=n)
#     result = [''] * n
#     for pi, texts in enumerate(other_lists):
#         positions = np.where(path_idxs == pi)[0]
#         if len(positions):
#             text_idxs = rng.integers(len(texts), size=len(positions))
#             for pos, ti in zip(positions, text_idxs):
#                 result[pos] = texts[ti]
#     return result

def derange_list_with_duplicates(original_list, rng):
    n = len(original_list)
    counts = Counter(original_list)
    most_common_count = max(counts.values())
    if most_common_count > n // 2:
        raise ValueError(
            f"Most frequent value appears {most_common_count}/{n} times; "
            "a strict derangement is impossible."
        )

    shuffled = original_list[:]
    rng.shuffle(shuffled)
    # Fix any position where the shuffled value matches the original
    for i in range(n):
        if shuffled[i] == original_list[i]:
            # Find a swap partner that won't create a new collision
            for j in range(i + 1, n):
                if shuffled[j] != original_list[i] and shuffled[i] != original_list[j]:
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                    break
    return shuffled




def process_dataset(df: pd.DataFrame, T_COL: str, Y_COL: str, TEXT_COL: str, data_name: str,rng: np.random.Generator,
                    out_dir: Path) -> None:
    # df = pd.read_csv(path)
    # t_col = T_COL if T_COL in df.columns else T_COL.lower()

    # df[t_col]   = pd.to_datetime(df[t_col]).astype('int64') // 10**9
    # df[TEXT_COL] = df[TEXT_COL].apply(clean_search_text)
    df = df[[T_COL, Y_COL, TEXT_COL]].sort_values(by=T_COL)

    # # Rename to canonical column names so downstream code is uniform
    # df = df.rename(columns={t_col: T_COL})

    split_idx = int(len(df) * 0.85)
    if split_idx < 400:
        print(f"Skipping {data_name} — train split has only {split_idx} rows (minimum 400)")
        return

    splits = [("train", df[:split_idx]), ("val", df[split_idx:])]

    for split_name, split_df in splits:
        points = [
            {
                "t":        int(row[T_COL]),
                "y":        float(row[Y_COL]),
                "category": "correct",
                "samples":  [{"text": row[TEXT_COL]}]
            }
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name} split")
            if isinstance(row[TEXT_COL], str)
        ]
        # Derange only valid (non-NaN) texts; map shuffled values back by position
        rows_list      = list(split_df.iterrows())
        valid_indices  = [i for i, (_, row) in enumerate(rows_list) if isinstance(row[TEXT_COL], str)]
        valid_texts    = [rows_list[i][1][TEXT_COL] for i in valid_indices]
        shuffled_texts = derange_list_with_duplicates(valid_texts, rng)
        irr_text_map   = {i: t for i, t in zip(valid_indices, shuffled_texts)}
        points += [
            {"t": int(row[T_COL]), "y": float(row[Y_COL]),
             "category": "irrelevant", "samples": [{"text": irr_text_map[i]}]}
            for i, (_, row) in enumerate(rows_list)
            if i in irr_text_map
        ]

        metadata = {
            "split":        split_name,
            "n_points":     len(split_df),
            "blank_points": (split_df[TEXT_COL].apply(lambda x: not isinstance(x, str))).sum() / len(split_df),
            "t":            df[T_COL].tolist(),
            "y":            df[Y_COL].tolist(),
        }

        out_path = out_dir / f"{split_name}.json"
        with open(out_path, 'w') as f:
            json.dump({'metadata': metadata, 'points': points},
                      f, indent=2, cls=_NumpyEncoder)
        print(f"Saved {len(points)} points → {out_path}")


def extract_info(text):
    region = re.search(r"region for prediction is (\w+)", text).group(1)
    start_date = re.search(r"start date of historical data was on ([\d-]+)", text).group(1)
    pred_date = re.search(r"date of prediction is on ([\d-]+)", text).group(1)
    return region, start_date, pred_date


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TimeMMD CSV files to benchmark split format."
    )
    # parser.add_argument("--input-dir",  required=True,
    #                     help="Directory containing CSV files")
    parser.add_argument("--output-dir", default = '/root/emma-code/mmtt-bench/datasets/fnf_dataset',
                        help="Directory to write <name>_train.json and <name>_val.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for irrelevant text sampling (default: 42)")
    args = parser.parse_args()

    # input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # data_paths = sorted(glob.glob(str(input_dir / "*.csv")))
    # AU_news_path = 'data/raw_news_data/AU-news/news_processed_data_all.json'
    AU_news_path = '/root/emma-code/From_News_to_Forecast/data/paired_time_series_news_training_data/AULF_train_data_2019-2020.json'
    bitcoin_news_path = '/root/emma-code/From_News_to_Forecast/data/raw_news_data/bitcoin_news.json'
    traffic_news_path = '/root/emma-code/From_News_to_Forecast/data/raw_news_data/traffic_news.json'

    bitcoin_ts_path = '/root/emma-code/From_News_to_Forecast/data/raw_time_series_data/bitcoin_daily.csv'
    exchange_ts_path = '/root/emma-code/From_News_to_Forecast/data/raw_time_series_data/Exchang_all_data_2018-2022_D_final.csv'
    traffic_ts_path = '/root/emma-code/From_News_to_Forecast/data/raw_time_series_data/traffic_hourly.csv'
    weather_ts_path = '/root/emma-code/From_News_to_Forecast/data/raw_time_series_data/weather_load_2019-2022.csv'

    data = {
        # fnfd
        'bitcoin': {"ts": bitcoin_ts_path, "text": bitcoin_news_path},
        'traffic': {"ts": traffic_ts_path, "text": traffic_news_path},
        'electricity' : {"ts": weather_ts_path, "text": AU_news_path},
        # # fnspid
        'delta_airlines': {"ts": '/root/emma-code/fnspid/full_history/DAL.csv', "text": '/root/emma-code/fnspid/DAL.json'},
        'ibm': {"ts": '/root/emma-code/fnspid/full_history/IBM.csv', "text": '/root/emma-code/fnspid/IBM.json'},
        'jpmorgan': {"ts": '/root/emma-code/fnspid/full_history/JPM.csv', "text": '/root/emma-code/fnspid/JPM.json'},
        'nvidia': {"ts": '/root/emma-code/fnspid/full_history/nvda.csv', "text": '/root/emma-code/fnspid/NVDA.json'},
        'pfizer': {"ts": '/root/emma-code/fnspid/full_history/PFE.csv', "text": '/root/emma-code/fnspid/PFE.json'},
        'tesla': {"ts": '/root/emma-code/fnspid/full_history/TSLA.csv', "text": '/root/emma-code/fnspid/TSLA.json'},
    }

    # Pre-load all text columns so irrelevant sampling can draw from other datasets
    # text_dict: dict[str, list] = {}
    # for data_name, data_dict in data.items():
    #     df = pd.read_json(data_dict["text"])
    #     if data_name == 'electricity':
    #         text_dict[data_name] = df['input'].tolist()
    #     elif data_name in ['delta_airlines', 'ibm', 'jpmorgan', 'nvidia', 'pfizer', 'tesla']:
    #         text_dict[data_name] = [f"News Article Title by {row['Publisher']}: {row['Article_title']}" for _,row in df.iterrows()]
    #     else:
    #         text_dict[data_name] = [f"News Article Title: {row['title']}, Article: {row['full_article']}" for _,row in df.iterrows()]

    rng = np.random.default_rng(seed=args.seed)

    # for path in [x['text'] for x in data.values()]:
    #     print(f"\nProcessing {path}")
    #     process_dataset(path, text_dict, rng, output_dir)

    T_COL = 'TIME'
    T_COL_TEXT = 'publication_time'
    TEXT_COL = 'TEXT'
    DATE_FORMAT = '%Y-%m-%d'

    for data_name, data_dict in data.items():
        print(f"\nProcessing {data_name}")
        df_ts = pd.read_csv(data_dict["ts"])
        df_text = pd.read_json(data_dict["text"])


        if data_name == 'bitcoin':
            Y_COL = 'VALUE'
            # has 18 different metrics across different rows
            df_ts = df_ts[df_ts['ID'] == 'price'].copy()
            df_text[TEXT_COL] = df_text.apply(lambda row: f"News Article Title: {row['title']}, Article: {row['full_article']}", axis=1)
        elif data_name == 'traffic':
            Y_COL = 'TRAFFIC VOLUME'
            df_ts = df_ts[df_ts['ID'] == 'T1'].copy()
            DATE_FORMAT = '%m/%d/%Y %H:%M'
            df_text[TEXT_COL] = df_text.apply(lambda row: f"News Article Title: {row['title']}, Article: {row['full_article']}", axis=1)
        elif data_name == 'electricity':
            T_COL = 't'
            Y_COL = 'TOTALDEMAND'
            TEXT_COL = 'input'
            DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
            df_text[['region', 'start_date', T_COL_TEXT]] = df_text['input'].apply(lambda x: pd.Series(extract_info(x)))
            df_ts[T_COL] = df_ts['SETTLEMENTDATE']
        elif data_name in ['delta_airlines', 'ibm', 'jpmorgan', 'nvidia', 'pfizer', 'tesla']:
            Y_COL = 'close'
            T_COL = 'date'
            T_COL_TEXT = 'Date'
            DATE_FORMAT = '%Y-%m-%d'
            df_text[TEXT_COL] = df_text.apply(lambda row: f"News Article Title from {row['Publisher']}: {row['Article_title']}", axis=1)
        else:
            raise NotImplementedError(f"Unknown dataset {data_name}")


        # match on date, as text only has date-level granularity
        df_ts['DATE'] = pd.to_datetime(df_ts[T_COL]).dt.date
        df_text['DATE'] =pd.to_datetime(df_text[T_COL_TEXT]).dt.date
        # TIME SERIES ONCE A DAY HERE
        # TEXT sometimes multiple times a day
        # group text as news articles on the same day together, one sample

        min_date = max(df_text['DATE'].min(), df_ts['DATE'].min())
        max_date = min(df_text['DATE'].max(), df_ts['DATE'].max())

        df_text = df_text.groupby('DATE')[TEXT_COL].apply(lambda x: '\n'.join(x)).reset_index()

        df = pd.merge(df_ts, df_text, left_on='DATE', right_on='DATE', how='left')
        df = df[df['DATE'].between(min_date, max_date)]

        # turn data into numerical sequence
        # this is more granular than date, true T_COL
        df[T_COL] = df[T_COL].apply(lambda x: int(datetime.strptime(str(x), DATE_FORMAT).timestamp() ))
        
        if data_name == 'electricity':
            dup_mask = df.duplicated(subset=[T_COL, 'State'], keep='first')
            df.loc[dup_mask, TEXT_COL] = np.nan

            for state in df['State'].unique():
                df_temp = df[df['State'] == state].copy()
                df_temp = df_temp.sort_values(by='SETTLEMENTDATE')
                df_temp[T_COL] = df_temp['SETTLEMENTDATE'].apply(lambda x: int(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp()))
                save_path = Path(output_dir)  / f"electricity_{state}"
                process_dataset(df, T_COL, Y_COL, TEXT_COL, data_name, rng, out_dir=save_path)

        else:
            save_path = Path(output_dir) / f"{data_name}"
            process_dataset(df, T_COL, Y_COL, TEXT_COL, data_name, rng, out_dir=save_path)
