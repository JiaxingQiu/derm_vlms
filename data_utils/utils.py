import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json  # Add this import at the top
from .tabular2text import *

class MIDASDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.data_dir / row['id_actual_filename']
        img = PILImage.open(img_path)#.convert('RGB')
        return {
            'image': img,
            # 'x': row[[col for col in row.index if col.startswith('x')]+['text_x_skintype', 'text_x_location']].to_dict(),
            'y': row[['y3', 'y16', 'y16_description', 'x_skintype', 'x_skincolor', 'x_skintone', 'x_location'] + ["text_full", "text_outcome", "text_y3", "text_y16", "text_x_skintype", "text_x_skincolor", "text_x_skintone", "text_x_location"]].to_dict(),
            'demo': row[[col for col in row.index if col.startswith('demo')]].to_dict(),
            'lesion': row[[col for col in row.index if col.startswith('lesion')]].to_dict(),
            'notes': row[[col for col in row.index if col.startswith('notes')]].to_dict(),
            'id': row[['id_patient', 'id_filename']].to_dict()
        }


def split_df(df, test_size=0.2, val_size=0.1, random_state=42):
    patients = df['id_patient'].unique()
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size/(1-test_size), random_state=random_state
    )
    train_df = df[df['id_patient'].isin(train_patients)]
    val_df = df[df['id_patient'].isin(val_patients)]
    test_df = df[df['id_patient'].isin(test_patients)]
    
    print(f"Train: {len(train_df)} images from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} images from {len(val_patients)} patients")
    print(f"Test: {len(test_df)} images from {len(test_patients)} patients")
    
    return train_df, val_df, test_df


# overall tabular processing
def process_tabular(data_dir):
    data_dir = Path(data_dir)
    df = pd.read_excel(data_dir / "release_midas.xlsx") 
    df = df.iloc[:, 1:]
    df = process_id(df)
    df = process_y(df)
    df = process_x_demo(df)
    df = process_x_lesion(df)
    df = process_x_notes(df)
    df = process_x_skintype(df)
    df = process_x_location(df)
    df = image_tabular_mapping(df, data_dir)
    text_cols = df.apply(lambda r: pd.Series(
        row_to_natural_text(r),
        index=["text_full","text_outcome","text_y3","text_y16","text_x_skintype","text_x_skincolor","text_x_skintone","text_x_location","text_demo","text_lesion"]
    ), axis=1)
    df = pd.concat([df, text_cols], axis=1)

    # Keep columns that start with id, y, demo, lesion, notes
    keep_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['id', 'y', 'x', 'demo', 'lesion', 'notes', 'text'])]
    df = df[keep_columns]
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns kept: {list(df.columns)}")
    return df

# 0. IDs
def process_id(df):
    df = df.copy()
    df['id_patient'] = df['midas_record_id'].astype(str)
    df['id_filename'] = df['midas_file_name'].astype(str)
    print(f"id_patient: {df['id_patient'].nunique()}")
    print(f"id_filename: {df['id_filename'].nunique()}")
    return df

# 1. Process outcome
def process_y(df):
    # More robust version that handles both string '0' and integer 0
    df = df.copy()
    df['midas_path'] = df['midas_path'].astype(str)
    df.loc[df['midas_path'].isna() | (df['midas_path'].astype(str) == '0'), 'midas_path'] = 'unknown'
    df['midas_path'] = df['midas_path'].str.replace('- ', '-')
    df['midas_path'] = df['midas_path'].str.lower()
    print(f"midas_path: {df['midas_path'].nunique()}")
    print(f"midas_path: {df['midas_path'].value_counts().to_dict()}")

    midas_path_mapping = {
        # Malignant categories
        "malignant-bcc": "Basal Cell Carcinoma - Most common type of skin cancer, slow-growing.",
        "malignant-melanoma": "Melanoma - Most dangerous form of skin cancer, can spread quickly.",
        "malignant-scc": "Squamous Cell Carcinoma - Second most common skin cancer.",
        "malignant-ak": "Actinic Keratosis - Precancerous lesion, can develop into SCC.",
        "malignant-sccis": "Squamous Cell Carcinoma In Situ - Early stage SCC, confined to epidermis.",
        "malignant-other": "Other Malignant - Other types of malignant skin lesions.",
        
        # Benign categories
        "benign-melanocytic nevus": "Melanocytic Nevus - Common mole, benign pigmented lesion.",
        "benign-other": "Other Benign - Other benign skin lesions not in specific categories.",
        "benign-seborrheic keratosis": "Seborrheic Keratosis - Common benign warty growth, barnacle of aging.",
        "benign-dermatofibroma": "Dermatofibroma - Benign fibrous tumor, often on legs.",
        "benign-hemangioma": "Hemangioma - Benign vascular tumor, blood vessel growth.",
        "benign-fibrous papule": "Fibrous Papule - Benign small skin growth, often on nose.",
        
        # Other/Uncertain categories
        "other-melanocytic lesion, possible re-excision (severe, spitz, aimp)": "Melanocytic Lesion - Uncertain melanocytic lesion requiring re-excision.",
        "other-non-neoplastic, inflammatory, infectious": "Non-neoplastic - Inflammatory or infectious skin conditions.",
        "melanocytic tumor, possible re-excision (severe, spitz, aimp)": "Melanocytic Tumor - Uncertain melanocytic tumor requiring re-excision.",
        "unknown": "Unknown - Unknown skin condition."
    }
    # ---- y: 16 outcomes ----
    df['y16_description'] = df['midas_path'].map(midas_path_mapping)
    print(f"y16_description: {df['y16_description'].nunique()}")
    # print(f"y16_description: {df['y16_description'].value_counts().to_dict()}")
    df['y16'] = df['y16_description'].str.split(' - ').str[0]
    print(f"y16: {df['y16'].nunique()}")
    print(f"y16: {df['y16'].value_counts().to_dict()}")
    # ---- y: 3 outcomes ----
    df['y3'] = "other"
    df.loc[df['midas_path'].str.startswith('malignant'), 'y3'] = "malignant"
    df.loc[df['midas_path'].str.startswith('benign'), 'y3'] = "benign"
    print(f"y3: {df['y3'].nunique()}")
    print(f"y3: {df['y3'].value_counts().to_dict()}")
    return df

# 2. Demographic features processing
def process_x_demo(df):
    df = df.copy()
    # Gender - binary categorical
    df['demo_gender'] = df['midas_gender'].str.lower().fillna('unknown')
    
    # Age - numerical, keep as is (already clean based on stats)
    df['demo_age'] = df['midas_age']
    # No changes needed for age
    
    # Fitzpatrick skin type - ordinal categorical (I-VI)
    df['demo_fitzpatrick_skintype'] = df['midas_fitzpatrick'].str.lower().fillna('unknown')
    
    # Melanoma history - binary categorical
    df['demo_melanoma_history'] = df['midas_melanoma'].str.lower().fillna('unknown')
    
    # Ethnicity - binary categorical
    df['demo_ethnicity'] = df['midas_ethnicity'].str.lower().fillna('unknown')
    
    # Race - multi-class categorical
    df['demo_race'] = df['midas_race'].str.lower().fillna('unknown')
    
    # Print cleaned value counts
    print("=== Demographics After Cleaning ===")
    print(f"demo_gender: {df['demo_gender'].value_counts().to_dict()}")
    print(f"demo_fitzpatrick_skintype: {df['demo_fitzpatrick_skintype'].value_counts().to_dict()}")
    print(f"demo_melanoma_history: {df['demo_melanoma_history'].value_counts().to_dict()}")
    print(f"demo_ethnicity: {df['demo_ethnicity'].value_counts().to_dict()}")
    print(f"demo_race: {df['demo_race'].value_counts().to_dict()}")
    print(f"Age range: {df['demo_age'].min():.1f} to {df['demo_age'].max():.1f}")
    
    return df

# 3. Lesion features processing
def process_x_lesion(df):
    df = df.copy()
    df['lesion_distance'] = df['midas_distance'].map({'6in': '6in', 'dscope': 'dscope', '1ft': '1ft', 'n/a - virtual': 'virtual'})
    df['lesion_location'] = df['midas_location']#.map({'chest': 'chest', 'back': 'back', 'other': 'other'})
    df['lesion_length_mm'] = df['length_(mm)']# as numerical
    df['lesion_width_mm'] = df['width_(mm)']# as numerical
    df['lesion_location'] = df['midas_location']
    df['lesion_location'] = df['lesion_location'].str.replace(r'^l ', 'left ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' l ', ' left ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' l$', 'left ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r'^r ', 'right ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' r ', 'right ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' r$', 'right ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r'^mid ', 'middle ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' mid ', ' middle ', regex=True)
    df['lesion_location'] = df['lesion_location'].str.replace(r' mid$', ' middle ', regex=True)
    # print a summary of the lesion features
    print("=== Lesion Features After Cleaning ===")
    print(f"lesion_distance: {df['lesion_distance'].value_counts().to_dict()}")
    print(f"lesion_location: {df['lesion_location'].nunique()}")
    print(f"lesion_length_mm: {df['lesion_length_mm'].min():.1f} to {df['lesion_length_mm'].max():.1f}")
    print(f"lesion_width_mm: {df['lesion_width_mm'].min():.1f} to {df['lesion_width_mm'].max():.1f}")
    return df

# 4. Notes features processing
def process_x_notes(df):
    df = df.copy()
    # midas_pathreport, clinical_impression_1, clinical_impression_2, clinical_impression_3
    df['notes_clinical_impression_1'] = df['clinical_impression_1'].str.lower()
    df['notes_clinical_impression_2'] = df['clinical_impression_2'].str.lower()
    df['notes_clinical_impression_3'] = df['clinical_impression_3'].str.lower()
    df['notes_pathreport'] = df['midas_pathreport'].str.lower()
    # # summarize the notes
    # print("=== Notes After Cleaning ===")
    # print(f"notes_clinical_impression_1: {df['notes_clinical_impression_1'].value_counts().to_dict()}")
    # print(f"notes_clinical_impression_2: {df['notes_clinical_impression_2'].value_counts().to_dict()}")
    # print(f"notes_clinical_impression_3: {df['notes_clinical_impression_3'].value_counts().to_dict()}")
    # print(f"notes_pathreport: {df['notes_pathreport'].value_counts().to_dict()}")
    return df
 
# 5. x predictor skintype (split into color and tone)
def process_x_skintype(df):
    # x_skintype is from demo_fitzpatrick_skintype
    df = df.copy()
    
    # Original combined mapping (kept for backwards compatibility)
    mapping_x1 = {
        "i pale white skin, blue/green eyes, blond/red hair": "light white skin",
        "ii fair skin, blue eyes": "white skin",
        "iii darker white skin": "dark white skin",
        "iv light brown skin": "light brown skin",
        "v brown skin": "brown skin",
        "vi dark brown or black skin": "dark brown skin",
        "unknown": "unknown",
    }
    
    # Skin color mapping: white vs brown
    color_mapping = {
        "i pale white skin, blue/green eyes, blond/red hair": "white",
        "ii fair skin, blue eyes": "white",
        "iii darker white skin": "white",
        "iv light brown skin": "brown",
        "v brown skin": "brown",
        "vi dark brown or black skin": "brown",
        "unknown": "unknown",
    }
    
    # Skin tone mapping: 3-level scale (light, medium, dark)
    tone_mapping = {
        "i pale white skin, blue/green eyes, blond/red hair": "light",
        "ii fair skin, blue eyes": "medium",
        "iii darker white skin": "dark",
        "iv light brown skin": "light",
        "v brown skin": "medium",
        "vi dark brown or black skin": "dark",
        "unknown": "unknown",
    }
    
    fitz_lower = df["demo_fitzpatrick_skintype"].str.lower()
    
    df["x_skintype"] = fitz_lower.map(mapping_x1).fillna("unknown")
    df["x_skincolor"] = fitz_lower.map(color_mapping).fillna("unknown")
    df["x_skintone"] = fitz_lower.map(tone_mapping).fillna("unknown")
    
    print(f"x_skintype: {df['x_skintype'].nunique()}")
    print(f"x_skintype: {df['x_skintype'].value_counts().to_dict()}")
    print(f"x_skincolor: {df['x_skincolor'].value_counts().to_dict()}")
    print(f"x_skintone: {df['x_skintone'].value_counts().to_dict()}")
    return df

# 6. x predictor lesion location (8 + other high-level regions)
def process_x_location(df):
    df = df.copy()

    def _clean_location(s):
        if pd.isna(s):
            return s
        s = s.lower().strip()

        # Remove directional / side descriptors
        s = re.sub(r'\b(left|right|upper|lower|middle|mid|central|medial|lateral|posterior|anterior|superior|inferior|proximal|distal)\b', '', s)

        # Remove words like "region", "area", "skin", "surface"
        s = re.sub(r'\b(region|area|skin|surface)\b', '', s)

        # Remove special chars, parentheses, extra spaces
        s = re.sub(r'[\(\)\[\]{}]', '', s)
        s = re.sub(r'[-_/]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()

        # Unify common variants (e.g., post auricular → postauricular)
        s = s.replace('post auricular', 'postauricular')
        s = s.replace('pre auricular', 'preauricular')
        s = s.replace('infra mammary', 'inframammary')
        s = s.replace('supra pubic', 'suprapubic')
        s = s.replace('buttox', 'buttock')

        # Simplify redundant phrases like "back (this is mislabeled...)" 
        s = re.sub(r'\b(back ).*', 'back', s)

        return s.strip()

    df['x_location'] = df['lesion_location'].apply(_clean_location)

    region_map = { # 8 categories
        # Head
        'cheek': 'head', 'forehead': 'head', 'temple': 'head',
        'eyelid': 'head', 'eyebrow': 'head', 'lip': 'head',
        'cutaneous lip': 'head', 'vermilion lip': 'head',
        'vermilion border': 'head', 'vermilion border of lip': 'head',
        'philtrum': 'head', 'chin': 'head', 'jaw': 'head',
        'jawline': 'head', 'malar cheek': 'head', 'mandible': 'head',
        'melolabia fold': 'head', 'nose': 'head', 'nasal bridge': 'head',
        'nasal dorsum': 'head', 'nasal sidewall': 'head', 'nasal ala': 'head',
        'nasal tip': 'head', 'nasal supratip': 'head', 'supra nasal tip': 'head',
        'alar crease': 'head', 'alar crease peri nasal': 'head', 
        'alar crease of nose': 'head', 'nose tip': 'head', 'canthus': 'head',
        'preauricular': 'head', 'postauricular': 'head', 'retro auricular': 'head',
        'preauricular cheek': 'head', 'side burn': 'head', 'sideburn': 'head',
        'occiput': 'head', 'frontal hairline': 'head', 'submental jaw': 'head',
        'frontal scalp': 'head', 'crown of scalp': 'head', 'vertex scalp': 'head',
        'scalp': 'head', 'scalp vertex': 'head', 'parietal scalp': 'head',
        'rt parietal': 'head', 'vertex': 'head', 'ear': 'head',
        'helix': 'head', 'helix of ear': 'head', 'antihelix': 'head',
        'antitragus': 'head', 'ear scapha': 'head', 'conchal bowl': 'head',
        'lobule of ear': 'head',

        # Neck
        'neck': 'neck', 'root of neck': 'neck', 'base of neck': 'neck',
        'midline neck': 'neck',

        # Torso (chest, back, abdomen, flank)
        'back': 'torso', 'low back': 'torso', 'chest': 'torso', 'low chest': 'torso',
        'midright chest': 'torso', 'abdomen': 'torso', 'flank': 'torso', 'ribcage': 'torso',
        'scapula': 'torso', 'clavicle': 'torso', 'breast': 'torso',
        'inframammary': 'torso', 'hypogastric': 'torso', 'suprapubic': 'torso',
        'umbilicus': 'torso', 'infirmary breast': 'torso',

        # Shoulder & Arm
        'shoulder': 'arm', 'deltoid': 'arm', 'arm': 'arm', 'forearm': 'arm',
        'dorsal forearm': 'arm', 'inner arm': 'arm', 'elbow': 'arm',
        'elbow joint': 'arm', 'axilla': 'arm', 'wrist': 'arm', 'hand radial': 'arm',

        # Hand & Fingers
        'hand': 'hand', 'dorsal hand': 'hand', 'hand ulnar': 'hand', 'finger': 'hand',
        '2nd finger': 'hand', '3rd finger': 'hand', '4th finger': 'hand', '5th finger': 'hand',
        'index finger': 'hand', 'thumb base': 'hand', 'pinky': 'hand', 'r4 digit': 'hand',

        # Leg (thigh, shin, calf, knee)
        'leg': 'leg', 'thigh': 'leg', 'shin': 'leg', 'l shin': 'leg', 'calf': 'leg',
        'post calf': 'leg', 'popliteal': 'leg', 'popliteal fossa': 'leg', 'knee': 'leg',

        # Foot & Ankle
        'foot': 'foot', 'dorsal foot': 'foot', 'heel': 'foot', 'plantar heel': 'foot',
        'plantar arch': 'foot', 'malleolus': 'foot', 'ankle': 'foot',
        '4th dorsal toe': 'foot', 'dorsal 2nd toe': 'foot', '3rd intertarsal': 'foot',

        # Pelvic / Groin / Buttock
        'groin': 'pelvic', 'inguinal fold': 'pelvic', 'inguinal crease': 'pelvic',
        'mons pubis': 'pelvic', 'buttock': 'pelvic', 'hip': 'pelvic',
        'superomedial thigh': 'pelvic', 'leg melanoma scar': 'pelvic',
    }
    
    df['x_location'] = df['x_location'].map(region_map).fillna('other')
    print(f"x_location: {df['x_location'].nunique()}")
    print(f"x_location: {df['x_location'].value_counts().to_dict()}")
    return df



def image_tabular_mapping(df, data_dir):
    """Create mapping between df filenames and actual file filenames"""
    
    mapping_data = []
    missing_files = []
    
    for idx, row in df.iterrows():
        df_filename = row['id_filename']
        base_name = df_filename.split('.')[0]  # Remove extension if present
        
        # Try different extensions
        actual_file = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            test_path = data_dir / (base_name + ext)
            if test_path.exists():
                actual_file = test_path.name
                break
        
        mapping_data.append({
            'id_filename': df_filename,
            'id_actual_filename': actual_file,
            'file_exists': actual_file is not None
        })
        
        if actual_file is None:
            missing_files.append(df_filename)
    
    # Create DataFrame
    mapping_df = pd.DataFrame(mapping_data)
    
    print(f"Found {mapping_df['file_exists'].sum()} files")  # Fixed: use 'file_exists' instead of 'id_actual_filename'
    print(f"Missing {len(missing_files)} files")
    if missing_files:
        print(f"Missing files: {missing_files[:10]}...")


    df = df.merge(mapping_df, on='id_filename', how='left')
    df = df[df['file_exists']].copy()
    print(f"Cleaned dataset: {len(df)} images")
    return df
    


