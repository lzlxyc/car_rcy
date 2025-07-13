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

    df = pd.read_json(data_path, lines=True).head(100)
    prompts = []
    labels = []
    preds = []
    reasons = []
    user_ids = []

    system = '你是专业的用户出行方式意图识别大师，请识别用户的下次出行方式，限定出行方式为以下5种：单车、助力车、顺风车、打车、租车。'
    post_prompt = "\n出行方式限定在以下5种：单车、助力车、顺风车、打车、租车, 请严格按照下面的字典格式输出：{'分析': 'xxx', '下次出行方式':xxx}"

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

        print(f">>> {i} {label=}|{pred=} | {reason=}")

    df_result = pd.DataFrame({
        'user_id': user_ids,
        'prompts': prompts,
        'labels': labels,
        'prediction': preds,
        'reason': reasons
    })
    df_result[['user_id', 'prediction', 'reason']].to_csv(f'../outputs/qwen-max-v2-{mode}-submit0713.csv', index=False)
    df_result.to_csv(f'../outputs/qwen-max-v2-{mode}-inference.csv', index=False)

    if mode == 'val':
        f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
        print("********************* val f1:", f1)


if __name__ == '__main__':
    aibox = AiBox(mode='api', model='qwplus')
    do_eval('../DATA/val_v2.jsonl')






