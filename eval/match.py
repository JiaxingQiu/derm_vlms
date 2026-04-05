"""Map free-text VLM diagnosis strings to the 16 canonical y16 labels.

Uses a hand-crafted synonym dictionary covering common dermatological
terms, abbreviations, and subtypes that VLMs typically produce.
"""

import re

Y16_LABELS = [
    "Basal Cell Carcinoma",
    "Melanoma",
    "Squamous Cell Carcinoma",
    "Actinic Keratosis",
    "Squamous Cell Carcinoma In Situ",
    "Other Malignant",
    "Melanocytic Nevus",
    "Other Benign",
    "Seborrheic Keratosis",
    "Dermatofibroma",
    "Hemangioma",
    "Fibrous Papule",
    "Melanocytic Lesion",
    "Non-neoplastic",
    "Melanocytic Tumor",
    "Unknown",
]

# Each key is a lowercase pattern; value is the y16 label.
# Ordered so more specific patterns come before more general ones.
_SYNONYM_MAP = {
    # --- Basal Cell Carcinoma ---
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "basal cell cancer": "Basal Cell Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "nodular bcc": "Basal Cell Carcinoma",
    "superficial bcc": "Basal Cell Carcinoma",
    "morpheaform basal cell": "Basal Cell Carcinoma",
    "pigmented basal cell": "Basal Cell Carcinoma",

    # --- Melanoma ---
    "melanoma": "Melanoma",
    "lentigo maligna melanoma": "Melanoma",
    "lentigo maligna": "Melanoma",
    "amelanotic melanoma": "Melanoma",
    "nodular melanoma": "Melanoma",
    "superficial spreading melanoma": "Melanoma",
    "melanoma in situ": "Melanoma",
    "acral melanoma": "Melanoma",

    # --- Squamous Cell Carcinoma ---
    "squamous cell carcinoma in situ": "Squamous Cell Carcinoma In Situ",
    "squamous cell carcinoma in-situ": "Squamous Cell Carcinoma In Situ",
    "scc in situ": "Squamous Cell Carcinoma In Situ",
    "scc in-situ": "Squamous Cell Carcinoma In Situ",
    "sccis": "Squamous Cell Carcinoma In Situ",
    "bowen's disease": "Squamous Cell Carcinoma In Situ",
    "bowen disease": "Squamous Cell Carcinoma In Situ",
    "bowenoid": "Squamous Cell Carcinoma In Situ",
    "squamous cell carcinoma": "Squamous Cell Carcinoma",
    "squamous cell cancer": "Squamous Cell Carcinoma",
    "scc": "Squamous Cell Carcinoma",
    "keratoacanthoma": "Squamous Cell Carcinoma",
    "invasive scc": "Squamous Cell Carcinoma",
    "cutaneous squamous cell carcinoma": "Squamous Cell Carcinoma",

    # --- Actinic Keratosis ---
    "actinic keratosis": "Actinic Keratosis",
    "actinic keratoses": "Actinic Keratosis",
    "solar keratosis": "Actinic Keratosis",
    "ak": "Actinic Keratosis",
    "hypertrophic actinic keratosis": "Actinic Keratosis",

    # --- Other Malignant ---
    "merkel cell carcinoma": "Other Malignant",
    "sebaceous carcinoma": "Other Malignant",
    "dermatofibrosarcoma": "Other Malignant",
    "kaposi sarcoma": "Other Malignant",
    "cutaneous lymphoma": "Other Malignant",
    "mycosis fungoides": "Other Malignant",

    # --- Melanocytic Nevus ---
    "melanocytic nevus": "Melanocytic Nevus",
    "melanocytic nevi": "Melanocytic Nevus",
    "nevus": "Melanocytic Nevus",
    "nevi": "Melanocytic Nevus",
    "mole": "Melanocytic Nevus",
    "common mole": "Melanocytic Nevus",
    "compound nevus": "Melanocytic Nevus",
    "junctional nevus": "Melanocytic Nevus",
    "intradermal nevus": "Melanocytic Nevus",
    "dysplastic nevus": "Melanocytic Nevus",
    "atypical nevus": "Melanocytic Nevus",
    "clark nevus": "Melanocytic Nevus",
    "blue nevus": "Melanocytic Nevus",
    "spitz nevus": "Melanocytic Nevus",
    "halo nevus": "Melanocytic Nevus",
    "congenital nevus": "Melanocytic Nevus",
    "reed nevus": "Melanocytic Nevus",

    # --- Seborrheic Keratosis ---
    "seborrheic keratosis": "Seborrheic Keratosis",
    "seborrheic keratoses": "Seborrheic Keratosis",
    "seborrheic wart": "Seborrheic Keratosis",
    "sk": "Seborrheic Keratosis",

    # --- Dermatofibroma ---
    "dermatofibroma": "Dermatofibroma",
    "fibrous histiocytoma": "Dermatofibroma",

    # --- Hemangioma ---
    "hemangioma": "Hemangioma",
    "haemangioma": "Hemangioma",
    "cherry angioma": "Hemangioma",
    "angioma": "Hemangioma",
    "vascular lesion": "Hemangioma",
    "pyogenic granuloma": "Hemangioma",
    "angiokeratoma": "Hemangioma",

    # --- Fibrous Papule ---
    "fibrous papule": "Fibrous Papule",
    "angiofibroma": "Fibrous Papule",

    # --- Melanocytic Lesion ---
    "melanocytic lesion": "Melanocytic Lesion",
    "atypical melanocytic proliferation": "Melanocytic Lesion",
    "atypical melanocytic lesion": "Melanocytic Lesion",

    # --- Melanocytic Tumor ---
    "melanocytic tumor": "Melanocytic Tumor",
    "melanocytic tumour": "Melanocytic Tumor",
    "spitzoid tumor": "Melanocytic Tumor",
    "spitzoid tumour": "Melanocytic Tumor",

    # --- Non-neoplastic ---
    "eczema": "Non-neoplastic",
    "dermatitis": "Non-neoplastic",
    "psoriasis": "Non-neoplastic",
    "lichen planus": "Non-neoplastic",
    "tinea": "Non-neoplastic",
    "fungal infection": "Non-neoplastic",
    "contact dermatitis": "Non-neoplastic",
    "atopic dermatitis": "Non-neoplastic",
    "stasis dermatitis": "Non-neoplastic",
    "nummular dermatitis": "Non-neoplastic",
    "nummular eczema": "Non-neoplastic",
    "discoid eczema": "Non-neoplastic",
    "cellulitis": "Non-neoplastic",
    "folliculitis": "Non-neoplastic",
    "impetigo": "Non-neoplastic",
    "rosacea": "Non-neoplastic",
    "lupus": "Non-neoplastic",
    "granuloma annulare": "Non-neoplastic",
    "morphea": "Non-neoplastic",
    "prurigo": "Non-neoplastic",
    "inflammatory": "Non-neoplastic",
    "infectious": "Non-neoplastic",

    # --- Other Benign ---
    "sebaceous hyperplasia": "Other Benign",
    "epidermal cyst": "Other Benign",
    "epidermoid cyst": "Other Benign",
    "lipoma": "Other Benign",
    "verruca": "Other Benign",
    "verruca vulgaris": "Other Benign",
    "common wart": "Other Benign",
    "wart": "Other Benign",
    "skin tag": "Other Benign",
    "acrochordon": "Other Benign",
    "keloid": "Other Benign",
    "scar": "Other Benign",
    "lentigo": "Other Benign",
    "solar lentigo": "Other Benign",
    "epidermal nevus": "Other Benign",
    "neurofibroma": "Other Benign",
    "trichilemmoma": "Other Benign",
    "porokeratosis": "Other Benign",
    "keratosis": "Other Benign",
    "benign keratosis": "Other Benign",
    "lichenoid keratosis": "Other Benign",
    "stucco keratosis": "Other Benign",
    "cyst": "Other Benign",
    "papilloma": "Other Benign",
    "arthropod bite": "Other Benign",
    "insect bite": "Other Benign",
}

# Pre-compile patterns: sort by length descending so longer (more specific)
# matches are tried first.
_COMPILED_PATTERNS = [
    (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
    for k, v in sorted(_SYNONYM_MAP.items(), key=lambda x: -len(x[0]))
]


def match_to_y16(diagnosis_text):
    """Map a single free-text diagnosis to a y16 label.

    Returns the matched y16 label string, or None if no match found.
    """
    if not diagnosis_text:
        return None
    text = diagnosis_text.strip()
    for pat, label in _COMPILED_PATTERNS:
        if pat.search(text):
            return label
    return None


def match_top3(diagnosis_list):
    """Map a list of extracted diagnosis strings to y16 labels.

    Returns a list of (original_text, matched_y16_or_None) tuples.
    """
    return [(d, match_to_y16(d)) for d in diagnosis_list]
