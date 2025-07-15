import pandas as pd
import numpy as np
from dateutil.parser import parse
from collections import Counter
from sklearn.model_selection import train_test_split
import datetime

# 1. 处理列表型字段（拆分并排序）
def process_list_fields(df, for_train=True):
    # if for_train:
    #     df = df[df['ord_cnt'] > 5].copy()

    # 拆分点击/订单列表
    df['clk_biz_list'] = df['clk_biz_list'].apply(lambda x: x.split(',') if pd.notna(x) else [])
    df['clk_date_time_list'] = df['clk_date_time_list'].apply(
        lambda x: [parse(t) for t in x.split(',')] if pd.notna(x) else []
    )
    df['ord_biz_list'] = df['ord_biz_list'].apply(lambda x: x.split(',') if pd.notna(x) else [])
    df['ord_date_time_list'] = df['ord_date_time_list'].apply(
        lambda x: [parse(t) for t in x.split(',')] if pd.notna(x) else []
    )

    # # 如果 ord_biz_list 为空，则用 clk_biz_list 填充
    # df['ord_biz_list'] = df.apply(
    #     lambda row: row['clk_biz_list'] if len(row['ord_biz_list']) == 0 else row['ord_biz_list'],
    #     axis=1
    # )
    # # 如果 ord_date_time_list 为空，则用 clk_date_time_list 填充
    # df['ord_date_time_list'] = df.apply(
    #     lambda row: row['clk_date_time_list'] if len(row['ord_date_time_list']) == 0 else row['ord_date_time_list'],
    #     axis=1
    # )
    # 按时间排序点击/订单序列
    def sort_by_time(biz_list, time_list):
        if len(biz_list) != len(time_list):
            min_len = min(len(biz_list), len(time_list))
            biz_list = biz_list[-min_len:]
            time_list = time_list[-min_len:]
        combined = sorted(zip(time_list, biz_list), key=lambda x: x[0])
        return [b for t, b in combined], [t for t, b in combined]

    df['sorted_clk_biz'], df['sorted_clk_time'] = zip(*df.apply(
        lambda row: sort_by_time(row['clk_biz_list'], row['clk_date_time_list']), axis=1
    ))

    df['sorted_ord_biz'], df['sorted_ord_time'] = zip(*df.apply(
        lambda row: sort_by_time(row['ord_biz_list'], row['ord_date_time_list']), axis=1
    ))
    return df


def analyze_travel_patterns(df):
    """
    对用户出行记录进行时间维度的统计分析，生成交通方式规律特征

    参数:
        df: 包含uid、sorted_ord_biz、sorted_ord_time字段的DataFrame

    返回:
        新增特征后的DataFrame
    """
    def time_split(row):
        if len(row['sorted_ord_time']) <=1:
            return row['sorted_ord_biz'], row['sorted_ord_time']

        if 'create_time' not in row.keys():
            end_day = row['sorted_ord_time'][-1]
        else:
            end_day = datetime.datetime.strptime(row['create_time'], "%Y-%m-%d %H:%M:%S.%f")  # 转换为datetime对象
        for i in range(len(row['sorted_ord_time'])):
            time_diff = (end_day - row['sorted_ord_time'][i]).days
            if time_diff < 45 or len(row['sorted_ord_time']) - i <= 15:
                row['sorted_ord_biz'] = row['sorted_ord_biz'][i:]
                row['sorted_ord_time'] = row['sorted_ord_time'][i:]
                break

        return row['sorted_ord_biz'], row['sorted_ord_time']

    df['sorted_ord_biz'], df['sorted_ord_time'] = zip(*df.apply(time_split, axis=1))

    # 初始化新特征列
    new_features = pd.DataFrame()
    new_features['user_id'] = df['user_id']

    # 1. 最常用的交通方式（整体频率最高）
    new_features['使用最多的出行方式'] = df.apply(
        lambda row: Counter(row['sorted_ord_biz']).most_common(1)[0][0]
        if row['sorted_ord_biz'] else None, axis=1
    )

    # 2. 最近常用的交通方式（取最近3次记录中频率最高的，不足3次则取全部）
    def get_recent_mode(row):
        if not row['sorted_ord_biz']:
            return None
        # 按时间排序（已排序但确保顺序），取最近3条

        combined = list(zip(row['sorted_ord_biz'], row['sorted_ord_time']))


        combined_sorted = sorted(combined, key=lambda x: x[1])  # 最近的在前
        recent_topK = 5 if len(combined_sorted) < 15 else 8
        recent = combined_sorted[-recent_topK:]  # 取最近3次
        recent_modes = [x[0] for x in recent]
        return Counter(recent_modes).most_common(1)[0][0]

    new_features['最近常用的出行方式'] = df.apply(get_recent_mode, axis=1)

    # 3. 周末常用的交通方式（周六日的记录）
    def get_weekend_mode(row):
        if not row['sorted_ord_biz']:
            return None
        # 筛选周末记录（5=周六，6=周日，注意：datetime.weekday()中周一为0，周日为6）
        weekend_modes = []
        for biz, time in zip(row['sorted_ord_biz'], row['sorted_ord_time']):
            if time.weekday() in [5, 6]:  # 周六或周日
                weekend_modes.append(biz)
        if not weekend_modes:
            return None
        return Counter(weekend_modes).most_common(1)[0][0]

    new_features['周末常用出行方式'] = df.apply(get_weekend_mode, axis=1)

    # 4. 非周末常用的交通方式（周一至周五的记录）
    def get_weekday_mode(row):
        if not row['sorted_ord_biz']:
            return None
        # 筛选工作日记录（0=周一至4=周五）
        weekday_modes = []
        for biz, time in zip(row['sorted_ord_biz'], row['sorted_ord_time']):
            if time.weekday() in [0, 1, 2, 3, 4]:  # 周一至周五
                weekday_modes.append(biz)
        if not weekday_modes:
            return None
        return Counter(weekday_modes).most_common(1)[0][0]

    new_features['工作日常用出出行方式'] = df.apply(get_weekday_mode, axis=1)

    # 5. 早高峰常用方式（7:00-9:00）
    def get_morning_peak_mode(row):
        if not row['sorted_ord_biz']:
            return None
        morning_modes = []
        for biz, time in zip(row['sorted_ord_biz'], row['sorted_ord_time']):
            if 7 <= time.hour < 10:  # 早高峰时段
                morning_modes.append(biz)
        if not morning_modes:
            return None
        return Counter(morning_modes).most_common(1)[0][0]

    new_features['早高峰常用出行方式'] = df.apply(get_morning_peak_mode, axis=1)

    # 6. 晚高峰常用方式（17:00-19:00）
    def get_evening_peak_mode(row):
        if not row['sorted_ord_biz']:
            return None
        evening_modes = []
        for biz, time in zip(row['sorted_ord_biz'], row['sorted_ord_time']):
            if 17 <= time.hour < 20:  # 晚高峰时段
                evening_modes.append(biz)
        if not evening_modes:
            return None
        return Counter(evening_modes).most_common(1)[0][0]

    new_features['晚高峰常用出行方式'] = df.apply(get_evening_peak_mode, axis=1)

    # 7. 各种出行方式的统计
    def get_all_mode_count(row):
        if not row['sorted_ord_biz']:
            return {}

        return dict(Counter(row['sorted_ord_biz']))

    new_features['各种出行方式次数统计'] = df.apply(get_all_mode_count, axis=1)

    # 8、最近三次出行方式
    def get_recent_three_mode(row):
        if not row['sorted_ord_biz']:
            return None
        return row['sorted_ord_biz'][-3:]

    new_features['最近三次出行方式'] = df.apply(get_recent_three_mode, axis=1)

    # 9、最近一次使用情况
    def get_last_mode_info(row):
        if not row['sorted_ord_biz']:
            return None
        last_mode = row['sorted_ord_biz'][-1]
        last_time = row['sorted_ord_time'][-1]
        return {'出行方式': last_mode, '出行时间': str(last_time)}

    new_features['最近一次使用情况'] = df.apply(get_last_mode_info, axis=1)
    # 合并原数据与新特征（按uid匹配）
    result = pd.merge(df, new_features.drop('user_id', axis=1), left_index=True, right_index=True)
    return result


def load_data(file_path='../DATA/训练数据.csv'):
    df = pd.read_csv(file_path)
    df = process_list_fields(df)
    df = analyze_travel_patterns(df)
    return df


def build_prompt(row):
    """将特征转化为Prompt"""
    # print(row['user_id'],row['各种出行方式次数统计'])
    prompt = f"用户信息：所在城市：{row['city_name']}\n"
    prompt += f"各种出行方式次数统计：使用{', '.join([f'{k}{v}次' for k,v in row['各种出行方式次数统计'].items()])}。\n"
    prompt += f"使用最多的出行方式：{row['使用最多的出行方式']}。\n"
    prompt += f"最近常用的出行方式：{row['最近常用的出行方式']}。\n"
    prompt += f"周末常用出行方式：{row['周末常用出行方式']}。\n"
    prompt += f"工作日常用出出行方式：{row['工作日常用出出行方式']}。\n"
    prompt += f"早高峰常用出行方式：{row['早高峰常用出行方式']}。\n"
    prompt += f"晚高峰常用出行方式：{row['晚高峰常用出行方式']}。\n"
    prompt += f"最近三次出行方式：{row['最近三次出行方式']}。\n"
    prompt += f"最近一次使用情况：{row['最近一次使用情况']}。\n"
    prompt += "请预测该用户下一次的出行方式：\n"
    return prompt

# 生成训练集（JSON格式，适配LLaMA-Factory）
def generate_dataset(features, out_file):
    def save_json(data, path):
        data.to_json(path, orient="records", force_ascii=False, lines=True)

    if 'label' not in features:
        features['label'] = ''

    dataset = []
    for _, row in features.iterrows():
        prompt = build_prompt(row)
        dataset.append({
            'user_id': row['user_id'],
            'label': row['label'],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": row['label']}
            ]
        })
    df = pd.DataFrame(dataset)

    if out_file == 'train':
        train, val = train_test_split(df, test_size=0.05, random_state=42, stratify=df['label'])
        save_json(train, f"../DATA/train_v3.jsonl")
        save_json(val, f"../DATA/val_v3.jsonl")
    else:
        save_json(df, f"../DATA/{out_file}_v3.jsonl")


if __name__ == '__main__':
    # df = load_data()
    # generate_dataset(df, 'train')


    df = load_data('../DATA/predict.csv')
    generate_dataset(df, 'test')
