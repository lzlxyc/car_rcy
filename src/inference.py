import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import f1_score


from llms import AiBox


def parse_output(output):
    # 去除Markdown代码块标记
    json_str = output.replace('```json\n', '').replace('\n```', '')

    # 解析JSON
    data = json.loads(json_str)

    return data


def do_eval(data_path='../DATA/val_v2.jsonl'):
    mode = 'val' if 'val' in data_path else 'test'

    df = pd.read_json(data_path, lines=True)
    prompts = []
    labels = []
    preds = []
    reasons = []
    user_ids = []

    system = '你是专业的用户出行方式意图识别大师，请识别用户的下次出行方式，限定出行方式为以下5种：单车、助力车、顺风车、打车、租车。'
    post_prompt = ("\n【出行方式限定】：\n"
                   "出行方式限定在以下5种：单车、助力车、顺风车、打车、租车;\n"
                   "【其他规则】：\n"
                   "1)如果用户最近使用的是‘租车’,则下次出行方式就使用‘租车’\n"
                   "2)如果用户没有任何出行记录，则下次出行方式就使用‘打车’\n"
                   "【输出格式】：\n"
                   "请严格按照下面的字典格式输出：{'分析': 'xxx', '下次出行方式':xxx}")

    for i in tqdm(range(len(df))):
        user_id = df.iloc[i].user_id
        prompt = df.iloc[i].conversations[0]['value']
        label = df.iloc[i].label

        response = aibox.chat(prompt + post_prompt, system=system)
        try:
            if 'json' in response:
                response = parse_output(response)
            else:
                response = eval(response)

            pred = response['下次出行方式']
            reason = response['分析']
        except Exception as e:
            pred = response
            reason = response
            print(e)

        user_ids.append(user_id)
        prompts.append(prompt)
        labels.append(label)
        preds.append(pred)
        reasons.append(reason)

        print(f">>> {i}|{user_id} {label=}|{pred=} | {reason=}")

    df_result = pd.DataFrame({
        'user_id': user_ids,
        'prompts': prompts,
        'labels': labels,
        'prediction': preds,
        'reason': reasons
    })
    if mode == 'val':
        f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
        print("********************* val f1:", f1)

    df_result[['user_id', 'prediction', 'reason']].to_csv(f'../outputs/qw-plus-v3-{mode}-submit0713.csv', index=False)
    df_result.to_csv(f'../outputs/qw-plus-v3-{mode}-inference.csv', index=False)




if __name__ == '__main__':
    aibox = AiBox(mode='api', model='qw-plus')
    do_eval('../DATA/test_v3.jsonl')






