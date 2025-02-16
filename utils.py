# -*- coding: utf-8 -*-
"""
Utility functions for news recommendation preprocessing and evaluation.
"""

import pandas as pd
import numpy as np
import logging
import time

# 设置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataUtils:
    """数据处理工具类，提供数据预处理和评估函数"""

    @staticmethod
    def get_user_item_time(click_df):
        """
        根据点击时间获取用户的点击文章序列 {user1: [(item1, time1), (item2, time2)...]...}
        """
        try:
            click_df = click_df.sort_values('click_timestamp')
            user_item_time_dict = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
                lambda x: list(zip(x['click_article_id'], x['click_timestamp']))
            ).to_dict()
            return user_item_time_dict
        except Exception as e:
            logging.error(f"Error in get_user_item_time: {e}")
            return None

    @staticmethod
    def get_item_user_time_dict(click_df):
        """
        根据时间获取商品被点击的用户序列 {item1: [(user1, time1), (user2, time2)...]...}
        """
        try:
            click_df = click_df.sort_values('click_timestamp')
            item_user_time_dict = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(
                lambda x: list(zip(x['user_id'], x['click_timestamp']))
            ).to_dict()
            return item_user_time_dict
        except Exception as e:
            logging.error(f"Error in get_item_user_time_dict: {e}")
            return None

    @staticmethod
    def get_hist_and_last_click(all_click):
        """
        获取当前数据的历史点击和最后一次点击
        """
        try:
            all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
            last_click_df = all_click.groupby('user_id').tail(1)

            def hist_func(user_df):
                return user_df if len(user_df) == 1 else user_df[:-1]

            hist_click_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

            return hist_click_df, last_click_df
        except Exception as e:
            logging.error(f"Error in get_hist_and_last_click: {e}")
            return None, None

    @staticmethod
    def get_item_info_dict(item_info_df):
        """
        获取文章 id 对应的基本属性，保存成字典的形式
        """
        try:
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

            item_type_dict = item_info_df.set_index('click_article_id')['category_id'].to_dict()
            item_words_dict = item_info_df.set_index('click_article_id')['words_count'].to_dict()
            item_created_time_dict = item_info_df.set_index('click_article_id')['created_at_ts'].to_dict()

            return item_type_dict, item_words_dict, item_created_time_dict
        except Exception as e:
            logging.error(f"Error in get_item_info_dict: {e}")
            return None, None, None

    @staticmethod
    def get_user_hist_item_info_dict(all_click):
        """
        获取用户历史点击的文章信息
        """
        try:
            user_hist_item_typs_dict = all_click.groupby('user_id')['category_id'].agg(set).to_dict()
            user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).to_dict()
            user_hist_item_words_dict = all_click.groupby('user_id')['words_count'].agg('mean').to_dict()

            all_click = all_click.sort_values('click_timestamp')
            user_last_item_created_time_dict = all_click.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).to_dict()

            return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict
        except Exception as e:
            logging.error(f"Error in get_user_hist_item_info_dict: {e}")
            return None, None, None, None

    @staticmethod
    def get_item_topk_click(click_df, k):
        """
        获取近期点击最多的文章
        """
        try:
            return click_df['click_article_id'].value_counts().index[:k].tolist()
        except Exception as e:
            logging.error(f"Error in get_item_topk_click: {e}")
            return None

    @staticmethod
    def metrics_hit_mrr(user_recall_items_dict, trn_last_click_df, topk=5):
        """
        计算 Hit Rate@TopK 和 MRR@TopK
        """
        try:
            last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
            user_num = len(user_recall_items_dict)

            hit_num = 0  # 命中数
            mrr_sum = 0  # MRR 分数累计

            for user, item_list in user_recall_items_dict.items():
                tmp_recall_items = [x[0] for x in item_list[:topk]]
                last_click_item = last_click_item_dict[user]

                if last_click_item in tmp_recall_items:
                    hit_num += 1

                for rank, item in enumerate(tmp_recall_items, start=1):
                    if item == last_click_item:
                        mrr_sum += 1.0 / rank
                        break

            hit_rate = round(hit_num / user_num, 5)
            mrr = round(mrr_sum / user_num, 5)

            logging.info(f"TopK: {topk} | Hit Rate: {hit_rate} | MRR: {mrr} | Users: {user_num}")
            return hit_rate, mrr
        except Exception as e:
            logging.error(f"Error in metrics_hit_mrr: {e}")
            return None, None


    # 节省内存的一个函数
    # 减少内存
    @staticmethod
    def reduce_mem(df):
        starttime = time.time()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.isnull(c_min) or pd.isnull(c_max):
                    continue
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                            100*(start_mem-end_mem)/start_mem,
                                                                                                            (time.time()-starttime)/60))
        return df