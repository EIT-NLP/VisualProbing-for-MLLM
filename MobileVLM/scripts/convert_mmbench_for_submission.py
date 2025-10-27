import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # args.annotation_file="/code/LLaVA-unet/LLaVA_orign/playground/data/eval/mmbench/mmbench_dev_20230712.tsv"
    # args.result_dir="/code/LLaVA-unet/LLaVA_orign/playground/data/eval/mmbench/answers/mmbench_dev_20230712"
    # args.upload_dir="/code/LLaVA-unet/LLaVA_orign/playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712"
    # args.experiment="CogVLM-finetune"
    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    # for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
    for pred in open(args.result_dir+".jsonl"):
        pred = json.loads(pred)
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']
        
    cur_df.to_excel(args.upload_dir + ".xlsx", index=False, engine='openpyxl')

    # cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
