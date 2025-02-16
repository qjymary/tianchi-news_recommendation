# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
import configparser
import collections
import logging
from tqdm import tqdm
from datetime import datetime
from utils import DataUtils
from gensim.models import Word2Vec
from annoy import AnnoyIndex

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取当前日期
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# 读取配置文件
config = configparser.ConfigParser()
config.read("config/config.ini")

# 获取参数
DATA_PATH = os.getenv("DATA_PATH", config["PROCESS"]["DATA_PATH"])
SAVE_PATH = os.getenv("SAVE_PATH", config["PROCESS"]["SAVE_PATH"])
MODE = os.getenv("MODE", config["PROCESS"]["MODE"])
METRIC_RECALL = config.getboolean("PROCESS", "METRIC_RECALL")

# 多路召回参数
TOPK = config.getint("COMBINE", "TOPK")

# 读取数据
logging.info("Loading data...")
all_click_df = pd.read_csv(os.path.join(SAVE_PATH, f"train_click_{MODE}_{CURRENT_DATE}.csv"))
if METRIC_RECALL:
    trn_hist_click_df, trn_last_click_df = DataUtils.get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

# 读取各路召回数据
user_multi_recall_dict = {
    'itemcf2_sim_itemcf_recall': {},
    'embedding_sim_item_recall': {},
    'word2vec_recall': {},
    'cold_start_recall': {}
}

weight_dict = {
    'itemcf2_sim_itemcf_recall': config.getfloat("COMBINE", "ITEMCF_WEIGHT"),
    'embedding_sim_item_recall': config.getfloat("COMBINE", "EMBEDDING_WEIGHT"),
    'word2vec_recall': config.getfloat("COMBINE", "WORD2VEC_WEIGHT"),
    'cold_start_recall': config.getfloat("COMBINE", "COLD_START_WEIGHT")
}

logging.info("Loading recall files...")
for method in user_multi_recall_dict.keys():
    recall_file = os.path.join(SAVE_PATH, f'{method}_{MODE}_{CURRENT_DATE}.pkl')
    if os.path.exists(recall_file):
        with open(recall_file, 'rb') as f:
            user_multi_recall_dict[method] = pickle.load(f)
        logging.info(f"Successfully loaded {method} recall data.")
    else:
        logging.warning(f"{method} recall file not found.")

# 召回结果合并
def combine_recall_results(user_multi_recall_dict, weight_dict, topk):
    final_recall_items_dict = {}

    def norm_user_recall_items_sim(sorted_item_list):
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        return [(item, (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0) for item, score in sorted_item_list]

    logging.info('Combining multiple recall results...')
    for method, user_recall_items in user_multi_recall_dict.items():
        recall_method_weight = weight_dict.get(method, 1.0)

        for user_id, sorted_item_list in user_recall_items.items():
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {user: sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk] for user, recall_item_dict in final_recall_items_dict.items()}

    final_recall_file = os.path.join(SAVE_PATH, f'final_recall_items_dict_{MODE}_{CURRENT_DATE}.pkl')
    pickle.dump(final_recall_items_dict_rank, open(final_recall_file, 'wb'))
    logging.info("Final recall items saved successfully.")

    return final_recall_items_dict_rank

final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, TOPK)

if METRIC_RECALL:
    logging.info("Evaluating final recall performance...")
    DataUtils.metrics_hit_mrr(final_recall_items_dict_rank, trn_last_click_df, topk=5)

logging.info("Multi-recall combination process completed successfully.")
