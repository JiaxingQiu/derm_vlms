import os, sys, torch

# --- setup ---
PROJECT_ROOT = '/scratch/jq2uw/derm_vlms'
SKINGPT_DIR = os.path.join(PROJECT_ROOT, 'skingpt')

if SKINGPT_DIR not in sys.path:
    sys.path.insert(0, SKINGPT_DIR)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

# --- load model ---
from model_skingpt4 import init_cfg, init_chat, chat_with_image
print('Loading model...')
cfg = init_cfg(gpu_id=0)
model, vis_processor, chat = init_chat(cfg)
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Total params:     {sum(p.numel() for p in model.parameters()):,}')

# --- load data ---
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(PROJECT_ROOT) / 'data'

df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data_share', 'midas_share.parquet'))
print(f'Loaded {len(df)} rows')
print(f'y3 distribution:\n{df["y3"].value_counts()}')

def resolve_img_path(p):
    p = str(p)
    if os.path.isfile(p):
        return p
    candidate = DATA_DIR / Path(p).name
    if candidate.is_file():
        return str(candidate)
    return p

df['image_path_resolved'] = df['image_path'].apply(resolve_img_path)
n_found = df['image_path_resolved'].apply(os.path.isfile).sum()
print(f'Resolved images: {n_found}/{len(df)} found')

# --- predict ---
question = 'Is the lesion malignant or benign, or other?'
results = []

print(f'Running predictions on {len(df)} images...')
print(f'Question: "{question}"')
print('-' * 60)

for i, row in tqdm(df.iterrows(), total=len(df)):
    uid = row['uid']
    img_path = row['image_path_resolved']

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'[SKIP] uid={uid}, cannot open {img_path}: {e}')
        continue

    response = chat_with_image(chat, image, question, temperature=0.0, remove_system=True)

    results.append({
        'uid': uid,
        'response': response,
    })

    if len(results) <= 3:
        print(f'[{len(results)}] uid={uid}  gt={row["y3"]}')
        print(f'    {response[:200]}')
        print()

print(f'Done. Collected {len(results)} predictions.')

# --- parse NLL probs and predicted label ---
import json

def parse_nll(response):
    idx = response.rfind('###NLL:')
    if idx == -1:
        return {'prob_malignant': None, 'prob_benign': None, 'prob_other': None, 'pred_label': None}
    try:
        nll = json.loads(response[idx + len('###NLL:'):].strip())
        probs = {lbl: nll[lbl]['prob'] for lbl in ['malignant', 'benign', 'other'] if lbl in nll}
        pred_label = max(probs, key=probs.get) if probs else None
        return {
            'prob_malignant': probs.get('malignant'),
            'prob_benign': probs.get('benign'),
            'prob_other': probs.get('other'),
            'pred_label': pred_label,
        }
    except Exception:
        return {'prob_malignant': None, 'prob_benign': None, 'prob_other': None, 'pred_label': None}

# --- save ---
results_dir = os.path.join(PROJECT_ROOT, 'skingpt', 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'raw_skingpt4_predictions.csv')

results_df = pd.DataFrame(results).set_index('uid')
nll_df = results_df['response'].apply(parse_nll).apply(pd.Series)
results_df = pd.concat([results_df, nll_df], axis=1)

# # strip NLL suffix from response, keep only free text
# results_df['response'] = results_df['response'].apply(lambda s: s.split('\n###NLL:')[0].strip())

print(f'\nPredicted label distribution:')
print(results_df['pred_label'].value_counts())
print(f'\nProb summary:')
print(results_df[['prob_malignant', 'prob_benign', 'prob_other']].describe())

results_df.to_csv(out_path)
print(f'\nSaved {len(results_df)} rows to {out_path}')
