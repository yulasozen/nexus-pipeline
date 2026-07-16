# ====================================================================
#  NEXUS Pipeline v1.0
#  NLP-driven Experienscape Understanding System
# ====================================================================
#  Reference implementation for the Master's thesis:
#
#    Özen, U. (2026). Automated Analysis of Cruise Passenger
#    Experiences: A Natural Language Processing Approach to Ship
#    Design Optimization. Aalto University, School of Engineering.
#
#  Developed within the CruiseFlex — Design for a Sustainable
#  Lifecycle project, funded by Business Finland (3179/31/2025).
#
#  Purpose:
#    An end-to-end NLP pipeline that converts unstructured cruise
#    passenger reviews into structured engineering KPIs. Each sentence
#    is classified into a 6x4 Design-Response Matrix — the five
#    environmental factors of Akter & Romanoff (2024) plus a sixth
#    column, Navigation & Exterior, proposed in the thesis — crossed
#    with the four internal response types of Pizam & Tasci (2019),
#    and sentiment-scored by a VADER + RoBERTa ensemble.
#
#  Research questions (thesis Section 1.4):
#    RQ1 — pipeline architecture              (Stages 1-7)
#    RQ2 — hybrid classification              (Stages 3-5)
#    RQ3 — ensemble sentiment reliability     (Stage 6)
#    RQ4 — engineering KPI synthesis          (Stage 7)
#
#  Data (thesis Chapter 4):
#    2,870 English reviews from TripAdvisor and CruiseCritic covering
#    Icon, Symphony, Harmony and Voyager of the Seas (RCI) and
#    Norwegian Prima (NCL). Raw review text is NOT redistributed with
#    this repository (platform ToS / GDPR); see the README for the
#    input format and a synthetic sample.
#
#  Tested environment: Kaggle Notebooks (2x T4 GPU); also runs on CPU.
#  Dependencies: requirements.txt  →  pip install -r requirements.txt
# ====================================================================

import subprocess

# ── STANDARD LIBRARY IMPORTS ─────────────────────────────────────
import os, re, time, warnings, json, pickle
from pathlib import Path
from datetime import datetime

# McNemar test: compares whether two models make different errors
from statsmodels.stats.contingency_tables import mcnemar

import numpy as np
import pandas as pd
import torch          # GPU availability check
import spacy          # Sentence segmentation
import openpyxl       # Excel output generation
import matplotlib.pyplot as plt

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# scikit-learn: classifier training and evaluation
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import (accuracy_score, cohen_kappa_score,
                                         classification_report, confusion_matrix)
from sklearn.model_selection     import (train_test_split, cross_val_score,
                                         StratifiedKFold)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise    import cosine_similarity
from scipy.sparse                import hstack, csr_matrix

# Encodes sentences into fixed-size dense vectors (384-dim)
from sentence_transformers import SentenceTransformer

# Transformer-based sentiment model fine-tuned on Twitter data
from transformers import pipeline as hf_pipeline

from langdetect import detect, LangDetectException
import anthropic   # Anthropic Claude API client

warnings.filterwarnings("ignore")

# ── API KEY MANAGEMENT ───────────────────────────────────────────
# API key is read securely from Kaggle Secrets.
# Falls back to environment variable if Secrets are unavailable.
# IMPORTANT: Never hard-code API keys directly in source code.
try:
    from kaggle_secrets import UserSecretsClient
    ANTHROPIC_API_KEY = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    print("✅ API key loaded from Kaggle Secrets")
except Exception:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    print("⚠️  Using environment variable for API key")

# ── FILE PATHS ───────────────────────────────────────────────────
# Paths are configurable via environment variables so the pipeline runs
# both locally (defaults below) and on Kaggle, e.g.:
#   NEXUS_DATA=/kaggle/input/<your-dataset>/master_reviews.csv
#   NEXUS_OUTPUT=/kaggle/working/outputs
# Input CSV columns: ship_name, review_body (optional: source) —
# see data/sample_reviews.csv for the expected format.
# MODEL_DIR stores trained classifier objects in pickle format.
DATA_PATH  = os.getenv("NEXUS_DATA", "data/reviews.csv")
OUTPUT_DIR = Path(os.getenv("NEXUS_OUTPUT", "outputs"))
MODEL_DIR  = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Fix random seed for reproducibility across runs
np.random.seed(42)

# ── GPU / CPU DETECTION ──────────────────────────────────────────
# Uses GPU (device=0) if CUDA is available, otherwise CPU (-1).
# GPU significantly speeds up embedding computation and RoBERTa inference.
DEVICE     = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "GPU ✅" if DEVICE == 0 else "CPU"

# ── PIPELINE CONFIGURATION ───────────────────────────────────────
# All thresholds and hyperparameters are centralised here.
# Changing these values controls pipeline behaviour without
# modifying logic elsewhere in the code.

MIN_SENTENCE_WORDS   = 5      # Sentences shorter than this are treated as noise
MAX_REVIEW_CHARS     = 4000   # Very long reviews are truncated (API token limit)
MIN_REVIEW_CHARS     = 80     # Very short reviews (single words, titles) are dropped
MIN_REVIEWS_FOR_SHIP = 150    # Safeguard for ship-level sheets; all five study vessels exceed this (thesis Table 3)
SENTIMENT_POS_THRESH = 0.10   # VADER compound score: >= +0.10 → positive
SENTIMENT_NEG_THRESH = -0.10  # VADER compound score: <= -0.10 → negative
N_ANNOTATE           = 2500   # Number of sentences sent to Claude for annotation
TEST_SIZE            = 0.20   # Train/test split ratio (20% held out for evaluation)
CONFIDENCE_THRESHOLD = 0.60   # Classifier predictions below this are flagged as uncertain
N_HUMAN_ANNOTATE     = 150    # Number of sentences sampled for human ground-truth validation
CV_FOLDS             = 5      # Number of folds for stratified cross-validation

# ── EXCEL COLOUR PALETTE ─────────────────────────────────────────
# Hex colour codes used in the output Excel workbook.
# Positive net scores are highlighted green, negative red.
NAVY  = "1A3A5C"; BLUE  = "2E6DA4"; DARK  = "404040"; DGRAY = "595959"
LBLUE = "DAE8FC"; GREEN = "D5E8D4"; RED   = "F8CECC"; LGRAY = "F5F5F5"
AMBER = "FFF2CC"; WHITE = "FFFFFF"

print(f"\n{'='*62}")
print(f"  NEXUS Pipeline v1.0 — NLP-driven Experienscape Understanding System")
print(f"  Akter & Romanoff (2024) taxonomy — domain-adapted classifier")
print(f"  {datetime.now().strftime('%d %B %Y, %H:%M')}")
print(f"  Device: {DEVICE_STR}")
print(f"{'='*62}\n")

# ╔══════════════════════════════════════════════════════════════╗
# ║  FRAMEWORK DEFINITIONS — 6x4 Design-Response Matrix          ║
# ║  Columns: Akter & Romanoff (2024); the sixth column,         ║
# ║           Navigation & Exterior, is a proposed extension     ║
# ║           introduced in this thesis (Section 2.3).           ║
# ║  Rows:    Pizam & Tasci (2019), via Akter & Romanoff (2024). ║
# ╚══════════════════════════════════════════════════════════════╝

# COLUMNS: Natural-language description of each environmental factor.
# These descriptions serve two purposes:
#   1. Provide context to Claude during annotation (Stage 3 prompt)
#   2. Act as anchor texts for embedding-based pre-sampling (Stage 3 strategy)
COLUMNS = {
    "Layout / Design Factors": (
        # Akter & Romanoff (2024) — thesis Table 1; cf. Bitner (1992); Kwortnik (2008)
        "Spatial layout, cabin size, room design, corridor arrangement, "
        "furniture placement, balcony, stateroom interior, pool deck layout, "
        "ship interior architecture, public space arrangement."
    ),
    "Ambient Factors": (
        # Akter & Romanoff (2024) — thesis Table 1; cf. Bitner (1992); Kwortnik (2008)
        "Temperature, air conditioning, heating. "
        "Lighting, natural light, brightness. "
        "Background music, noise levels, acoustics. "
        "Air quality, ventilation, odour, scent."
    ),
    "Social Factors": (
        # Akter & Romanoff (2024) — thesis Table 1; cf. Kwortnik (2008)
        "Staff friendliness, crew attitude, customer service quality. "
        "Crowding, queuing, congestion. "
        "Boarding and embarkation process, check-in. "
        "Fellow passenger behaviour, social atmosphere."
    ),
    "Product / Service Factors": (
        # Akter & Romanoff (2024) — thesis Table 1
        "Food quality, dining service, buffet, menu, restaurant experience. "
        "Beverages, bar service. "
        "Spa, massage, gym, fitness, wellness services. "
        "Wi-Fi, room service, retail shops."
    ),
    "On-board Enjoyment Factors": (
        # Akter & Romanoff (2024) — thesis Table 1; cf. Kwortnik (2008)
        "Entertainment shows, theatre, Broadway, aqua show, ice show. "
        "Casino, rock climbing, ice skating, flowrider, water slides. "
        "Kids club, dance classes, trivia, onboard games and activities."
    ),
    "Navigation & Exterior": (
        # Proposed extension introduced in this thesis (Özen, 2026) — Section 2.3
        "Ship exterior, hull, promenade deck, open decks. "
        "Shore excursions, port destinations, island visits. "
        "Sea days, ocean views, cruise itinerary, ports of call. "
        "Safety drills, muster station, lifeboat."
    ),
}

# ROWS: Definitions of passenger internal response types.
# These are used directly in the Claude annotation prompt.
ROWS = {
    "Cognitive": (
        # Pizam & Tasci (2019): beliefs, categorisations, symbolic meanings
        "The passenger expresses a belief, opinion, evaluation, comparison "
        "or recommendation based on rational assessment."
    ),
    "Affective": (
        # Pizam & Tasci (2019): pleasure/displeasure, arousal
        "The passenger expresses a clear emotion: joy, love, excitement, "
        "happiness, anger, sadness, disappointment or frustration."
    ),
    "Physiological": (
        # Pizam & Tasci (2019): movement, pain, comfort/discomfort
        "The passenger describes a physical body sensation: comfort, "
        "discomfort, pain, seasickness, noise disturbance, "
        "temperature feeling, cramped space or fatigue."
    ),
    "Behavioral": (
        # Bitner (1992): approach/avoidance behaviours
        "The passenger describes an action taken or intended: "
        "waiting in a queue, complaining, booking, recommending, "
        "avoiding, returning or cancelling."
    ),
}

COLUMNS_ORDER = list(COLUMNS.keys())  # Preserves insertion order for consistent indexing
ROWS_ORDER    = list(ROWS.keys())

# ── MARITIME-DOMAIN VADER LEXICON ────────────────────────────────
# VADER is trained on general English text and misjudges several
# cruise-specific terms. The adjustment values below correct known
# under-scoring of negative domain vocabulary.
# Source: corpus inspection + Kwortnik (2008)
# Values are in the [-1, +1] range; negative values reduce sentiment score.
MARITIME_NEGATIVES = {
    "vibration":    -0.30,   # Ship vibration — consistently negative experience
    "cramped":      -0.35,   # Cabin space complaints
    "smelly":       -0.35,
    "musty":        -0.30,   # Old/damp smell — cabin quality indicator
    "rust":         -0.35,
    "sewage":       -0.50,   # Serious infrastructure issue
    "shaking":      -0.25,
    "mold":         -0.40,
    "broken":       -0.30,
    "leaking":      -0.35,
    "stuffy":       -0.20,
    "filthy":       -0.45,
    "stained":      -0.30,
    "outdated":     -0.20,
    "overpriced":   -0.30,
    "rude":         -0.40,
    "understaffed": -0.35,
    "ignored":      -0.35,
    "crowded":      -0.20,
    "noisy":        -0.20,
}

# ── EXCEL HELPER FUNCTIONS ───────────────────────────────────────
# Reusable cell formatting functions for the output Excel workbook.

def xl_border():
    """Returns a thin grey border applied to all four sides of a cell."""
    t = Side(style="thin", color="CCCCCC")
    return Border(left=t, right=t, top=t, bottom=t)

def xl_header_cell(ws, row, col, value, fill_color=None, size=10):
    """Creates a header cell: white bold text on a coloured background."""
    c = ws.cell(row=row, column=col)
    c.value = value
    c.font  = Font(bold=True, color=WHITE, size=size, name="Arial")
    c.fill  = PatternFill("solid", fgColor=fill_color or NAVY)
    c.alignment = Alignment(horizontal="center", vertical="center",
                            wrap_text=True)
    c.border = xl_border()
    return c

def xl_data_cell(ws, row, col, value, fill_color=LGRAY,
                 text_color="000000", bold=False):
    """Creates a data cell; caller supplies background/text colour for
    positive (green) or negative (red) highlighting."""
    c = ws.cell(row=row, column=col)
    c.value = value
    c.font  = Font(bold=bold, color=text_color, size=10, name="Arial")
    c.fill  = PatternFill("solid", fgColor=fill_color)
    c.alignment = Alignment(horizontal="center", vertical="center",
                            wrap_text=True)
    c.border = xl_border()
    return c


# ════════════════════════════════════════════════════════════════
# STAGE 1 — DATA LOADING & CLEANING
# ════════════════════════════════════════════════════════════════
# Goal: Build a clean, English-only, deduplicated review dataset
#       from the raw CSV file.
#
# Steps:
#   1. Load CSV and standardise column names
#   2. Length filter (drop very short / truncate very long reviews)
#   3. Language detection — remove non-English reviews
#   4. Exact and near-duplicate removal
#   5. Ship name normalisation (CruiseCritic URL format → clean name)
# ════════════════════════════════════════════════════════════════
print("━"*62)
print("  STAGE 1 — Data loading & cleaning")
print("━"*62)

def is_english(text):
    """Returns True if langdetect identifies the text as English."""
    try:    return detect(str(text)) == "en"
    except: return False

def clean_text(text):
    """
    Basic text normalisation:
    - Collapses multiple whitespace characters into a single space
    - Strips HTML tags (some CruiseCritic reviews contain HTML)
    - Normalises typographic dashes and smart quotes to ASCII equivalents
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("–","-").replace("—","-")
    text = text.replace("\u2019","'").replace("\u201c",'"').replace("\u201d",'"')
    return text

df_raw = pd.read_csv(DATA_PATH)
print(f"  Loaded: {len(df_raw):,} rows")

# Standardise column names to lowercase with underscores
df_raw.columns = df_raw.columns.str.lower().str.replace(" ","_")
if "source" not in df_raw.columns:
    df_raw["source"] = "unknown"

df = df_raw.dropna(subset=["review_body"]).copy()
df["review_body"] = df["review_body"].astype(str).apply(clean_text)

# Drop reviews that are too short to contain meaningful content
df = df[df["review_body"].str.len() >= MIN_REVIEW_CHARS]

# Truncate very long reviews to respect embedding and API token limits
long_mask = df["review_body"].str.len() > MAX_REVIEW_CHARS
df.loc[long_mask, "review_body"] = df.loc[long_mask, "review_body"].str[:MAX_REVIEW_CHARS]

# Remove non-English reviews (langdetect)
print("  Language detection...")
df["is_english"] = df["review_body"].apply(is_english)
df = df[df["is_english"]].drop(columns=["is_english"])

# Remove exact duplicates and near-duplicates (first 200 chars as key)
before = len(df)
df = df.drop_duplicates(subset="review_body")
df["_key"] = df["review_body"].str.strip().str[:200]
df = df.drop_duplicates(subset="_key").drop(columns=["_key"])
df = df.reset_index(drop=True)
print(f"  Removed {before - len(df)} duplicates")

# Normalise CruiseCritic's URL-based ship names to clean, consistent labels
SHIP_MAP = {
    "Royal Caribbean Harmony of the Seas Deck Plans & Reviews": "Harmony Of The Seas",
    "Royal Caribbean Symphony of the Seas Deck Plans & Reviews": "Symphony Of The Seas",
    "Icon of the Seas Deck Plans & Reviews": "Icon Of The Seas",
    "Norwegian Cruise Line Norwegian Prima Deck Plans & Reviews": "Norwegian Prima",
    "Voyager of the Seas Deck Plans & Reviews": "Voyager Of The Seas",
}
df["ship_name"] = df["ship_name"].replace(SHIP_MAP).str.strip().str.title()

print(f"\n  Clean dataset: {len(df):,} reviews")
for ship, n in df["ship_name"].value_counts().items():
    # Ships below MIN_REVIEWS_FOR_SHIP are excluded from ship-level comparisons
    flag = " ⚠️" if n < MIN_REVIEWS_FOR_SHIP else ""
    print(f"    {ship:<35} {n:>4}{flag}")
print(f"  ✅ Stage 1 complete")


# ════════════════════════════════════════════════════════════════
# STAGE 2 — SENTENCE SPLITTING & EMBEDDING COMPUTATION
# ════════════════════════════════════════════════════════════════
# Goal: Move from review level to sentence level, and encode every
#       sentence as a fixed-size (384-dim) semantic vector.
#
# Why sentence level?
#   A single review can address multiple topics simultaneously.
#   Sentence-level classification provides finer granularity.
#   Example: "The food was excellent. The cabin, however, was small."
#   → two sentences, two different columns.
#   NOTE: contradictory clauses inside ONE sentence (e.g. "The cabin
#   was small but the food was excellent") are NOT split further —
#   a known limitation discussed in thesis Section 5.3.
#
# Why all-MiniLM-L6-v2?
#   - Produces compact 384-dim embeddings; fast and memory-efficient
#   - Output is L2-normalised → cosine similarity = dot product
#   - From the Sentence-BERT family; strong on semantic similarity tasks
#   - The sampling strategy in Stage 3 depends on these embeddings
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 2 — Sentence splitting & embedding")
print("━"*62)

# spaCy is used for rule-based sentence boundary detection
# en_core_web_sm: small English model, sufficient for segmentation
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm", "-q"])
nlp = spacy.load("en_core_web_sm")
print("  ✅ spaCy loaded")

print("  Loading all-MiniLM-L6-v2...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("  ✅ Embedder loaded")

print(f"\n  Splitting sentences...")
rows = []
for _, rev in df.iterrows():
    doc = nlp(rev["review_body"])
    for sent in doc.sents:
        text = sent.text.strip()
        # Sentences shorter than MIN_SENTENCE_WORDS are discarded as noise
        if len(text.split()) >= MIN_SENTENCE_WORDS:
            rows.append({"ship_name": rev["ship_name"],
                         "source":    rev["source"],
                         "sentence":  text})

df_sents = pd.DataFrame(rows).reset_index(drop=True)
print(f"  Total sentences: {len(df_sents):,}")

# Encode all sentences in batches
# normalize_embeddings=True ensures L2 normalisation for cosine similarity
print("  Computing sentence embeddings...")
sent_embeddings = embedder.encode(
    df_sents["sentence"].tolist(),
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,
    device="cuda" if DEVICE == 0 else "cpu",
    convert_to_numpy=True,
)

# Encode the column anchor texts
# Used in Stage 3 to identify which sentences belong near each column
print("  Computing column anchor embeddings...")
col_embeddings = embedder.encode(
    list(COLUMNS.values()),
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# Persist embeddings to disk — reused in Stage 4 without recomputation
np.save(str(OUTPUT_DIR / "sent_embeddings.npy"), sent_embeddings)
np.save(str(OUTPUT_DIR / "col_embeddings.npy"),  col_embeddings)
df_sents.to_csv(str(OUTPUT_DIR / "sentences.csv"), index=False)

# Assign each sentence to its nearest column via cosine similarity.
# NOTE: this is used ONLY for the sampling strategy in Stage 3,
# not as the final classification result.
sim_matrix = cosine_similarity(sent_embeddings, col_embeddings)
df_sents["emb_col"]     = [COLUMNS_ORDER[i] for i in np.argmax(sim_matrix, axis=1)]
df_sents["emb_col_sim"] = np.max(sim_matrix, axis=1).round(4)

print(f"  ✅ Stage 2 complete")


# ════════════════════════════════════════════════════════════════
# STAGE 3 — CLAUDE ANNOTATION (2,500 sentences)
# ════════════════════════════════════════════════════════════════
# Goal: Create a domain-adapted labeled dataset to train the classifier.
#
# Why use Claude as annotator?
#   - Zero-shot general-purpose classifiers lack maritime terminology,
#     and rule-based tools lack semantic context (thesis Section 1.3)
#   - Manual annotation of 2,500 sentences is not practical at this scale
#   - Claude can apply the Akter & Romanoff (2024) framework directly from the prompt,
#     acting as a consistent expert annotator
#   - This approach is referred to as "LLM-supervised domain adaptation"
#     in the NLP literature (Pan & Yang, 2010; Wan et al., 2024; Tan et al., 2024)
#
# Sampling strategy:
#   ~416 sentences are drawn from each of the 6 columns (balanced).
#   Within each column, a mix of low and high cosine-similarity sentences
#   is selected, ensuring boundary cases are included in training.
#   This prevents the classifier from only learning easy, clear-cut examples.
#
# Error handling:
#   If Claude returns a malformed JSON or an out-of-vocabulary label,
#   the sentence is skipped. Progress is saved every 100 annotations
#   so the run can be resumed after interruption without re-annotating.
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 3 — Claude annotation (2,500 sentences)")
print("━"*62)

CACHE_PATH = str(OUTPUT_DIR / "annotation_cache.csv")

# Draw ~416 sentences from each column
N_PER_COL = N_ANNOTATE // len(COLUMNS_ORDER)
sample_frames = []
for col in COLUMNS_ORDER:
    mask   = df_sents["emb_col"] == col
    subset = df_sents[mask].copy().sort_values("emb_col_sim")
    # 60% low-confidence (ambiguous / boundary cases) + 40% high-confidence (clear cases)
    n_low  = min(int(N_PER_COL * 0.6), len(subset))
    n_high = min(N_PER_COL - n_low, len(subset) - n_low)
    sample_frames.append(pd.concat([subset.head(n_low), subset.tail(n_high)]))

sample_df = (pd.concat(sample_frames)
               .drop_duplicates(subset="sentence")
               .sample(frac=1, random_state=42)   # shuffle
               .reset_index(drop=True)
               .head(N_ANNOTATE))

print(f"  Sample: {len(sample_df):,} (~{N_PER_COL} per column)")
print(f"  Est. cost: ~${len(sample_df)*350/1_000_000*0.80:.2f}")

# System prompt passed to Claude for every annotation request.
# Includes full framework definitions and explicit disambiguation rules
# for borderline cases (e.g. food + staff → Product/Service, not Social).
SYSTEM_PROMPT = f"""You are an expert annotator for a cruise passenger experience research project at Aalto University.

Classify each cruise review sentence according to Akter and Romanoff (2024) and Pizam and Tasci (2019).

COLUMNS (choose ONE):
{chr(10).join(f"{i+1}. {col}: {desc}" for i,(col,desc) in enumerate(COLUMNS.items()))}

ROWS (choose ONE):
{chr(10).join(f"{i+1}. {row}: {desc}" for i,(row,desc) in enumerate(ROWS.items()))}

RULES:
- Food/dining mentions → Product / Service Factors
- Physical sensation (pain, heat, noise, motion) → Physiological
- Explicit emotion word (love, angry, excited, devastated) → Affective
- Opinion/evaluation/comparison/recommendation → Cognitive
- Action taken (waited, complained, booked, avoided) → Behavioral

OUTPUT: JSON only, no other text:
{{"column": "<exact name>", "row": "<exact name>", "explanation": "<one sentence>"}}"""

def annotate_sentence(sentence, client):
    """
    Sends a single sentence to Claude Haiku for annotation.
    Haiku was chosen for speed and cost efficiency.
    Parses the JSON response; returns None on any error.
    """
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",   # Claude Haiku 4.5 — cited as claude-haiku-4-5 in thesis Section 3.5
            max_tokens=150,   # Short JSON response; 150 tokens is sufficient
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f'Classify:\n\n"{sentence}"'}]
        )
        # The model occasionally wraps JSON in ```json ... ``` fences — strip them
        raw    = re.sub(r"```json|```", "", msg.content[0].text).strip()
        result = json.loads(raw)
        # Validate that the returned labels exist in the framework
        assert result["column"] in COLUMNS_ORDER
        assert result["row"]    in ROWS_ORDER
        return result
    except Exception:
        return None  # Sentence is skipped; counted in errors

# Load existing cache to resume after interruption
annotated = {}
if Path(CACHE_PATH).exists():
    cache_df = pd.read_csv(CACHE_PATH)
    for _, row in cache_df.iterrows():
        annotated[int(row["idx"])] = {
            "claude_col":  row["claude_col"],
            "claude_row":  row["claude_row"],
            "explanation": row["explanation"],
        }
    print(f"  ♻️  Loaded {len(annotated):,} from cache")

if ANTHROPIC_API_KEY != "YOUR_API_KEY_HERE":
    client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    pending = [i for i in sample_df.index if i not in annotated]
    errors  = 0
    buf     = []

    print(f"\n  Annotating {len(pending)} sentences...")
    for i, idx in enumerate(pending):
        result = annotate_sentence(sample_df.at[idx, "sentence"], client)
        if result:
            annotated[idx] = {"claude_col":  result["column"],
                               "claude_row":  result["row"],
                               "explanation": result["explanation"]}
            buf.append({"idx":         idx,
                        "sentence":    sample_df.at[idx, "sentence"],
                        "claude_col":  result["column"],
                        "claude_row":  result["row"],
                        "explanation": result["explanation"]})
        else:
            errors += 1

        # Flush buffer to disk every 100 annotations (interruption tolerance)
        if len(buf) >= 100:
            chunk = pd.DataFrame(buf)
            hdr   = not Path(CACHE_PATH).exists()
            chunk.to_csv(CACHE_PATH, mode="a", header=hdr, index=False)
            buf   = []
            print(f"  {i+1}/{len(pending)} | errors: {errors} | 💾 saved")

        time.sleep(0.15)   # Gentle rate limiting to avoid API overload

    # Flush remaining buffer
    if buf:
        chunk = pd.DataFrame(buf)
        hdr   = not Path(CACHE_PATH).exists()
        chunk.to_csv(CACHE_PATH, mode="a", header=hdr, index=False)

    print(f"\n  Annotated: {len(annotated):,} | Errors: {errors}")

# Apply labels back to sample_df; drop sentences without a valid annotation
sample_df["claude_col"]  = [annotated.get(i,{}).get("claude_col")  for i in sample_df.index]
sample_df["claude_row"]  = [annotated.get(i,{}).get("claude_row")  for i in sample_df.index]
sample_df["explanation"] = [annotated.get(i,{}).get("explanation","") for i in sample_df.index]

labeled = sample_df.dropna(subset=["claude_col","claude_row"]).copy()
labeled.to_csv(str(OUTPUT_DIR / "labeled_dataset.csv"), index=False)

print(f"\n  Labeled dataset: {len(labeled):,}")
print(f"  Column distribution (Claude labels):")
for col in COLUMNS_ORDER:
    n = (labeled["claude_col"]==col).sum()
    print(f"    {col:<35} {n:>4}  ({n/len(labeled)*100:.1f}%)")
print(f"  Row distribution (Claude labels):")
for row in ROWS_ORDER:
    n = (labeled["claude_row"]==row).sum()
    print(f"    {row:<15} {n:>4}  ({n/len(labeled)*100:.1f}%)")
print(f"  ✅ Stage 3 complete")


# ════════════════════════════════════════════════════════════════
# STAGE 3b — HUMAN ANNOTATION SUBSET (Ground Truth Validation)
# ════════════════════════════════════════════════════════════════
# Goal: Validate Claude's annotations against independent human labels.
#
# Why is this step necessary?
#   Claude's labels are "pseudo-labels" — they serve as a proxy for ground
#   truth but are not equivalent to human expert annotation.
#   Without human validation, the concern arises that the classifier is
#   simply learning to mimic Claude rather than the underlying framework.
#   Comparing against human labels addresses this directly.
#
# Workflow:
#   1. human_annotation_task.csv is generated (25 sentences per column)
#   2. A human annotator manually labels the file (single-annotator
#      check — see thesis Sections 4.2 and 5.3)
#   3. Saved as human_labels.csv in the outputs folder
#   4. On the next pipeline run, Cohen's Kappa (Claude vs Human) is computed
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 3b — Human annotation (ground truth validation)")
print("━"*62)
print(f"""
  ⚠️  ACTION REQUIRED — Human Ground Truth
  ─────────────────────────────────────────
  To validate Claude's annotations, {N_HUMAN_ANNOTATE} sentences need
  human labeling. The task file has been saved for you.

  FORMAT (CSV with columns):
    sentence, human_col, human_row

  WHERE to label:
    File: {OUTPUT_DIR}/human_annotation_task.csv
    Fill in the human_col and human_row columns,
    then save as human_labels.csv in the same folder.

  Column options : {COLUMNS_ORDER}
  Row options    : {ROWS_ORDER}

  If human_labels.csv exists, this stage will compute agreement metrics.
""")

human_task_path  = str(OUTPUT_DIR / "human_annotation_task.csv")
human_label_path = str(OUTPUT_DIR / "human_labels.csv")

if not Path(human_task_path).exists():
    # Stratified sample: 25 sentences per column for balanced coverage
    human_frames = []
    for col in COLUMNS_ORDER:
        sub = labeled[labeled["claude_col"]==col].sample(
            min(25, (labeled["claude_col"]==col).sum()), random_state=42)
        human_frames.append(sub)
    human_task = pd.concat(human_frames)[
        ["sentence","claude_col","claude_row","explanation"]]
    human_task["human_col"] = ""   # To be filled in by researcher
    human_task["human_row"] = ""
    human_task.to_csv(human_task_path, index=False)
    print(f"  📋 Task file saved: human_annotation_task.csv")

# If human labels are available, compute agreement metrics
if Path(human_label_path).exists():
    human_df = pd.read_csv(human_label_path)
    human_df = human_df.dropna(subset=["human_col","human_row"])

    # Cohen's Kappa between Claude and human annotator
    ch_col_kappa = cohen_kappa_score(
        human_df["claude_col"], human_df["human_col"],
        labels=COLUMNS_ORDER)
    ch_row_kappa = cohen_kappa_score(
        human_df["claude_row"], human_df["human_row"],
        labels=ROWS_ORDER)

    print(f"\n  ── CLAUDE vs HUMAN AGREEMENT ──")
    print(f"  Column Kappa : {ch_col_kappa:.3f}")
    print(f"  Row Kappa    : {ch_row_kappa:.3f}")
    print(f"  N sentences  : {len(human_df)}")

    col_human_acc = accuracy_score(human_df["human_col"], human_df["claude_col"])
    row_human_acc = accuracy_score(human_df["human_row"], human_df["claude_row"])
    print(f"  Claude accuracy vs human: Col={col_human_acc:.1%}  Row={row_human_acc:.1%}")
else:
    print(f"\n  ⚠️  human_labels.csv not found — skipping human validation")
    ch_col_kappa = None
    ch_row_kappa = None


# ════════════════════════════════════════════════════════════════
# STAGE 4 — CLASSIFIER TRAINING + ABLATION STUDY
# ════════════════════════════════════════════════════════════════
# Goal: Train a domain-adapted classifier on the Claude-labeled dataset
#       and measure the contribution of each feature set.
#
# Feature engineering:
#   Combined feature set = TF-IDF + Sentence Embeddings
#
#   TF-IDF (n-gram range 1–3):
#     - Captures word and phrase frequency patterns
#     - Effective for domain-specific n-grams ("spa treatment", "buffet line")
#     - max_features=20,000: retains the most informative n-grams
#     - sublinear_tf=True: log TF scaling reduces the weight of very frequent terms
#
#   Sentence Embeddings (all-MiniLM-L6-v2, 384 dim):
#     - Captures semantic similarity beyond exact word overlap
#     - "the cabin was cramped" ≈ "stateroom space was very limited" (thesis Section 3.6)
#     - Handles paraphrases that TF-IDF would miss
#
#   Feature combination (hstack):
#     - Sparse TF-IDF matrix + dense embeddings → hybrid sparse matrix
#     - The two feature sets are complementary
#
# Model choice — Logistic Regression:
#   - Well-suited for high-dimensional sparse data (TF-IDF compatible)
#   - Produces calibrated probability scores (needed for confidence threshold)
#   - class_weight="balanced": automatically compensates for class imbalance
#     (the labelled set is naturally imbalanced across categories — thesis Section 3.6)
#   - Interpretable: coefficients reveal which n-grams drive each class
#
# Evaluation design:
#   - 80/20 stratified train/test split (column distribution preserved)
#   - 5-fold cross-validation: reduces dependence on a single split
#   - Cohen's Kappa: accounts for class imbalance (more conservative than accuracy)
#   - McNemar test: tests whether the accuracy difference between models is
#     statistically significant (not due to chance)
#   - Per-class F1: reveals performance on underrepresented categories
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 4 — Classifier training + ablation study")
print("━"*62)

labeled = pd.read_csv(str(OUTPUT_DIR / "labeled_dataset.csv"))
print(f"  Dataset: {len(labeled):,} sentences")

# Retrieve embeddings computed in Stage 2 (avoids recomputation)
df_sents_tmp  = pd.read_csv(str(OUTPUT_DIR / "sentences.csv"))
sent_emb_full = np.load(str(OUTPUT_DIR / "sent_embeddings.npy"))
sent_to_emb   = {row["sentence"]: sent_emb_full[i]
                 for i, row in df_sents_tmp.iterrows()}
labeled_embs  = np.array([sent_to_emb.get(s, np.zeros(384))
                           for s in labeled["sentence"]])

# Build three feature sets for the ablation study
tfidf   = TfidfVectorizer(ngram_range=(1,3), max_features=20_000,
                          min_df=2, sublinear_tf=True)
# NOTE: the vectoriser is fitted on the FULL labelled set before the
# train/test split. This mildly leaks vocabulary statistics into the
# test fold (~2-3 pp optimism for the TF-IDF baseline, less for the
# full model) and is acknowledged in thesis Section 5.3. It is kept
# as-run so the published numbers can be reproduced exactly.
X_tfidf = tfidf.fit_transform(labeled["sentence"])
X_emb   = csr_matrix(labeled_embs)
X_full  = hstack([X_tfidf, X_emb])                   # combined feature set

print(f"  Features: TF-IDF={X_tfidf.shape[1]:,} + Emb={384} = {X_full.shape[1]:,}")

# Stratified split: column class distribution is preserved in both sets
X_tr_full, X_te_full, idx_tr, idx_te = train_test_split(
    X_full, labeled.index, test_size=TEST_SIZE,
    random_state=42, stratify=labeled["claude_col"])
X_tr_tfidf = X_tfidf[idx_tr]; X_te_tfidf = X_tfidf[idx_te]
X_tr_emb   = X_emb[idx_tr];   X_te_emb   = X_emb[idx_te]
train_df   = labeled.loc[idx_tr]
test_df    = labeled.loc[idx_te]
print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

def train_eval(X_train, X_test, y_train_col, y_test_col,
               y_train_row, y_test_row, name):
    """
    Trains column and row classifiers on the given feature set,
    evaluates on the test set, and returns a metrics dictionary.
    Used for all three ablation conditions (TF-IDF, Emb, Full).
    """
    col_clf = LogisticRegression(C=1.0, max_iter=1000,
                                  class_weight="balanced",
                                  random_state=42, solver="lbfgs")
    col_clf.fit(X_train, y_train_col)
    col_pred  = col_clf.predict(X_test)
    col_acc   = accuracy_score(y_test_col, col_pred)
    col_kappa = cohen_kappa_score(y_test_col, col_pred, labels=COLUMNS_ORDER)

    row_clf = LogisticRegression(C=1.0, max_iter=1000,
                                  class_weight="balanced",
                                  random_state=42, solver="lbfgs")
    row_clf.fit(X_train, y_train_row)
    row_pred  = row_clf.predict(X_test)
    row_acc   = accuracy_score(y_test_row, row_pred)
    row_kappa = cohen_kappa_score(y_test_row, row_pred, labels=ROWS_ORDER)

    print(f"\n  [{name}]")
    print(f"  Column: {col_acc:.1%} acc  κ={col_kappa:.3f}")
    print(f"  Row   : {row_acc:.1%} acc  κ={row_kappa:.3f}")
    return {"name": name,
            "col_acc": col_acc, "col_kappa": col_kappa,
            "row_acc": row_acc, "row_kappa": row_kappa,
            "col_clf": col_clf, "row_clf": row_clf,
            "col_pred": col_pred, "row_pred": row_pred}

# Ablation study: run all three feature conditions
print(f"\n  ── ABLATION STUDY (baseline comparison) ──")
res_tfidf = train_eval(X_tr_tfidf, X_te_tfidf,
                       train_df["claude_col"], test_df["claude_col"],
                       train_df["claude_row"], test_df["claude_row"],
                       "TF-IDF only")
res_emb   = train_eval(X_tr_emb, X_te_emb,
                       train_df["claude_col"], test_df["claude_col"],
                       train_df["claude_row"], test_df["claude_row"],
                       "Embeddings only")
res_full  = train_eval(X_tr_full, X_te_full,
                       train_df["claude_col"], test_df["claude_col"],
                       train_df["claude_row"], test_df["claude_row"],
                       "TF-IDF + Embeddings (full)")

baseline_results = [res_tfidf, res_emb, res_full]

# The combined model (TF-IDF + Embeddings) is selected as the final classifier
col_clf   = res_full["col_clf"]
row_clf   = res_full["row_clf"]
col_acc   = res_full["col_acc"]
col_kappa = res_full["col_kappa"]
row_acc   = res_full["row_acc"]
row_kappa = res_full["row_kappa"]
col_pred  = res_full["col_pred"]
row_pred  = res_full["row_pred"]

# 5-fold cross-validation: provides a more stable accuracy estimate
# StratifiedKFold preserves class proportions in each fold
print(f"\n  ── {CV_FOLDS}-FOLD CROSS-VALIDATION ──")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

col_cv = cross_val_score(
    LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                       random_state=42, solver="lbfgs"),
    X_full, labeled["claude_col"], cv=skf, scoring="accuracy")
row_cv = cross_val_score(
    LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                       random_state=42, solver="lbfgs"),
    X_full, labeled["claude_row"], cv=skf, scoring="accuracy")

# mean ± std format indicates both performance level and stability
print(f"  Column CV: {col_cv.mean():.1%} ± {col_cv.std():.1%}")
print(f"  Row    CV: {row_cv.mean():.1%} ± {row_cv.std():.1%}")

# McNemar test: tests whether the full model and TF-IDF-only model
# make statistically different errors (not just a random improvement).
# H0: the two models make errors on the same sentences.
# p < 0.05: reject H0 → adding embeddings significantly changes error patterns.
print(f"\n  ── MCNEMAR TEST (Full vs TF-IDF only) ──")
def mcnemar_test(pred_a, pred_b, true_labels):
    """
    Computes the McNemar test statistic comparing two classifiers.
    pred_a: predictions of the full model
    pred_b: predictions of the TF-IDF baseline
    Returns p-value (lower = more significant difference).
    """
    correct_a = np.array(pred_a) == np.array(true_labels)
    correct_b = np.array(pred_b) == np.array(true_labels)
    b = (~correct_a & correct_b).sum()   # B correct, A wrong
    c = (correct_a & ~correct_b).sum()   # A correct, B wrong
    table = [[0, b], [c, 0]]
    try:
        result = mcnemar(table, exact=False, correction=True)
        return result.pvalue
    except Exception:
        if b + c == 0: return 1.0
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        return 1 - chi2_dist.cdf(chi2, df=1)

y_test_col = test_df["claude_col"].values
y_test_row = test_df["claude_row"].values

p_col = mcnemar_test(res_full["col_pred"], res_tfidf["col_pred"], y_test_col)
p_row = mcnemar_test(res_full["row_pred"], res_tfidf["row_pred"], y_test_row)

print(f"  Column: Full vs TF-IDF  p={p_col:.4f} "
      f"{'✅ significant (embeddings contribute)' if p_col < 0.05 else '⚠️ not significant'}")
print(f"  Row   : Full vs TF-IDF  p={p_row:.4f} "
      f"{'✅ significant (embeddings contribute)' if p_row < 0.05 else '⚠️ not significant'}")

# Detailed per-class results (precision, recall, F1)
print(f"\n  ── FULL MODEL DETAILED RESULTS ──")
print(f"\n  COLUMN CLASSIFIER:")
print(classification_report(y_test_col, col_pred,
                             labels=COLUMNS_ORDER, zero_division=0))
print(f"  ROW CLASSIFIER:")
print(classification_report(y_test_row, row_pred,
                             labels=ROWS_ORDER, zero_division=0))

# Error analysis: identifies where and why the classifier fails.
# Provides concrete evidence for the limitations section of the thesis.
print(f"\n  ── ERROR ANALYSIS ──")
test_df_copy = test_df.copy()
test_df_copy["col_pred"]    = col_pred
test_df_copy["row_pred"]    = row_pred
test_df_copy["col_correct"] = (test_df_copy["claude_col"] == col_pred)
test_df_copy["row_correct"] = (test_df_copy["claude_row"] == row_pred)

# Most frequently confused column pairs
col_errors = test_df_copy[~test_df_copy["col_correct"]]
col_confusion_pairs = (col_errors["claude_col"] + " → " +
                       col_errors["col_pred"]).value_counts().head(5)
print(f"\n  Top column confusions (true → predicted):")
for pair, n in col_confusion_pairs.items():
    print(f"    {pair}  ({n}×)")

row_errors = test_df_copy[~test_df_copy["row_correct"]]
row_confusion_pairs = (row_errors["claude_row"] + " → " +
                       row_errors["row_pred"]).value_counts().head(5)
print(f"\n  Top row confusions (true → predicted):")
for pair, n in row_confusion_pairs.items():
    print(f"    {pair}  ({n}×)")

# Example error sentences help explain why confusions occur
print(f"\n  Example column errors:")
for _, row in col_errors.head(5).iterrows():
    print(f"    [{row['claude_col'][:20]} → {row['col_pred'][:20]}]")
    print(f"    \"{row['sentence'][:80]}\"")

test_df_copy.to_csv(str(OUTPUT_DIR / "test_set_results.csv"), index=False)

# Persist trained models in pickle format for reuse in Stage 5
pickle.dump(col_clf, open(str(MODEL_DIR/"col_classifier.pkl"), "wb"))
pickle.dump(row_clf, open(str(MODEL_DIR/"row_classifier.pkl"), "wb"))
pickle.dump(tfidf,   open(str(MODEL_DIR/"tfidf.pkl"),          "wb"))

print(f"\n  ✅ Stage 4 complete")
print(f"     Column: {col_acc:.1%} acc  κ={col_kappa:.3f}  CV={col_cv.mean():.1%}±{col_cv.std():.1%}")
print(f"     Row   : {row_acc:.1%} acc  κ={row_kappa:.3f}  CV={row_cv.mean():.1%}±{row_cv.std():.1%}")


# ════════════════════════════════════════════════════════════════
# STAGE 5 — FULL DATASET CLASSIFICATION + CONFIDENCE ANALYSIS
# ════════════════════════════════════════════════════════════════
# Goal: Apply the trained classifier to all 50,398 sentences and
#       report prediction confidence for transparency.
#
# Confidence threshold (CONFIDENCE_THRESHOLD = 0.60):
#   Logistic Regression outputs softmax probabilities for each class.
#   The maximum class probability is used as the confidence score.
#   Predictions below 0.60 are flagged as "uncertain".
#
# Two output versions are produced:
#   1. Full dataset  — all 50k sentences (uncertain ones flagged, not removed)
#   2. High-confidence subset — only sentences where BOTH classifiers
#      exceed the threshold simultaneously
#   Comparing the two versions in the thesis demonstrates how confidence
#   filtering affects the KPI findings.
#
# Design note: uncertain sentences are NOT excluded from the main analysis.
#   All sentences are classified; uncertainty is reported transparently.
#   Logistic-regression probabilities are reasonably well calibrated
#   out of the box (Niculescu-Mizil & Caruana, 2005); thesis Section 3.8
#   discusses the multi-class caveat on reading these scores.
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 5 — Full dataset classification + confidence analysis")
print("━"*62)

# Build features for all 50k sentences
# tfidf.transform (not fit!) — the vocabulary is fixed from Stage 4
X_tfidf_all = tfidf.transform(df_sents["sentence"])
X_all_full  = hstack([X_tfidf_all, csr_matrix(sent_embeddings)])

# Predict labels and extract confidence scores
col_probs = col_clf.predict_proba(X_all_full)
col_preds = col_clf.predict(X_all_full)
col_confs = col_probs.max(axis=1)   # max softmax probability = confidence

row_probs = row_clf.predict_proba(X_all_full)
row_preds = row_clf.predict(X_all_full)
row_confs = row_probs.max(axis=1)

# Attach results to the sentence dataframe
df_sents["pred_col"]       = col_preds
df_sents["col_confidence"] = col_confs.round(4)
df_sents["pred_row"]       = row_preds
df_sents["row_confidence"] = row_confs.round(4)
df_sents["col_uncertain"]  = col_confs < CONFIDENCE_THRESHOLD
df_sents["row_uncertain"]  = row_confs < CONFIDENCE_THRESHOLD

# High-confidence subset: both classifiers above threshold
high_conf_mask = ((col_confs >= CONFIDENCE_THRESHOLD) &
                  (row_confs >= CONFIDENCE_THRESHOLD))
df_sents_hc = df_sents[high_conf_mask].copy()

n_high = high_conf_mask.sum()
print(f"  Total sentences         : {len(df_sents):,}")
print(f"  High confidence (≥{CONFIDENCE_THRESHOLD}): {n_high:,} ({n_high/len(df_sents)*100:.1f}%)")
print(f"  Low confidence          : {(~high_conf_mask).sum():,} ({(~high_conf_mask).mean()*100:.1f}%)")

print(f"\n  ── FULL DATASET — column distribution ──")
for col in COLUMNS_ORDER:
    n = (df_sents["pred_col"]==col).sum()
    print(f"    {col:<35} {n:>6,}  ({n/len(df_sents)*100:.1f}%)")

print(f"\n  ── HIGH-CONFIDENCE SUBSET — column distribution ──")
for col in COLUMNS_ORDER:
    n = (df_sents_hc["pred_col"]==col).sum()
    print(f"    {col:<35} {n:>6,}  ({n/len(df_sents_hc)*100:.1f}%)")

print(f"\n  Row distribution (full dataset):")
for row in ROWS_ORDER:
    n = (df_sents["pred_row"]==row).sum()
    print(f"    {row:<15} {n:>6,}  ({n/len(df_sents)*100:.1f}%)")

df_sents.to_csv(str(OUTPUT_DIR / "classified_rows.csv"), index=False)
df_sents_hc.to_csv(str(OUTPUT_DIR / "classified_rows_highconf.csv"), index=False)
print(f"\n  ✅ Stage 5 complete")


# ════════════════════════════════════════════════════════════════
# STAGE 6 — SENTIMENT ANALYSIS (VADER + RoBERTa Ensemble)
# ════════════════════════════════════════════════════════════════
# Goal: Assign a sentiment label (positive / neutral / negative) and
#       a confidence level to every sentence. Directly addresses RQ3.
#
# Two-model ensemble rationale:
#
#   VADER (Valence Aware Dictionary and sEntiment Reasoner):
#     - Rule-based, lexicon-driven model
#     - Extended with a maritime-domain lexicon for cruise-specific terms
#     - Fast, transparent, and fully explainable
#     - Weakness: limited context understanding; cannot detect irony
#
#   RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest):
#     - Transformer-based, context-aware model
#     - Fine-tuned on Twitter data — suited to short informal text
#     - Cruise reviews share a similar informal, short-sentence style
#     - Weakness: may misinterpret cruise-domain vocabulary
#
#   Ensemble decision logic:
#     - Both agree          → final label = agreed label  (confidence: high)
#     - VADER is neutral    → defer to RoBERTa             (confidence: medium)
#     - RoBERTa is neutral  → defer to VADER               (confidence: medium)
#     - Direct conflict     → defer to RoBERTa             (confidence: low)
#
# RQ3 answer:
#   VADER–RoBERTa agreement rate and Cohen's Kappa are computed overall
#   and per column. Columns with lower agreement indicate topics where
#   sentiment is more ambiguous — a substantive methodological finding.
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 6 — Sentiment analysis (VADER + RoBERTa ensemble)")
print("━"*62)

vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    """
    Computes VADER compound score and applies maritime-domain corrections.
    Compound score is in [-1, +1]; thresholds determine the final label.
    """
    c = vader.polarity_scores(str(text))["compound"]
    # Apply domain-specific adjustments for terms VADER underscores
    for word, adj in MARITIME_NEGATIVES.items():
        if word in text.lower():
            c = max(-1.0, min(1.0, c + adj))
    if   c >= SENTIMENT_POS_THRESH: return "positive", round(c,4)
    elif c <= SENTIMENT_NEG_THRESH: return "negative", round(c,4)
    else:                           return "neutral",  round(c,4)

print("  Loading RoBERTa...")
try:
    roberta = hf_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=DEVICE,
        truncation=True,    # Sentences exceeding 512 tokens are truncated
        max_length=512,
        batch_size=64)      # Batch size tuned for GPU memory
    roberta_ok = True
    print("  ✅ RoBERTa loaded")
except Exception:
    roberta_ok = False
    print("  ⚠️  RoBERTa unavailable — falling back to VADER only")

def rob_label(label):
    """Maps RoBERTa's raw label string to the standard 3-class scheme."""
    label = label.lower()
    if "positive" in label: return "positive"
    if "negative" in label: return "negative"
    return "neutral"

print(f"  Running VADER on {len(df_sents):,} sentences...")
vader_res = [vader_sentiment(s) for s in df_sents["sentence"]]
df_sents["vader_label"] = [r[0] for r in vader_res]
df_sents["vader_score"] = [r[1] for r in vader_res]

if roberta_ok:
    print("  Running RoBERTa...")
    rb_res = roberta(df_sents["sentence"].tolist())
    df_sents["roberta_label"] = [rob_label(r["label"]) for r in rb_res]
    df_sents["roberta_score"] = [round(r["score"],4)   for r in rb_res]
else:
    df_sents["roberta_label"] = df_sents["vader_label"]
    df_sents["roberta_score"] = df_sents["vader_score"]

def ensemble(v, r):
    """
    Ensemble rule combining VADER (v) and RoBERTa (r) labels.
    Returns (final_label, confidence_level).
    """
    if v == r:           return v, "high"
    elif v == "neutral": return r, "medium"
    elif r == "neutral": return v, "medium"
    else:                return r, "low"    # Conflict: RoBERTa preferred

ens = [ensemble(v, r) for v, r in
       zip(df_sents["vader_label"], df_sents["roberta_label"])]
df_sents["sentiment"]            = [e[0] for e in ens]
df_sents["sentiment_confidence"] = [e[1] for e in ens]

# Propagate sentiment labels to the high-confidence subset as well
df_sents_hc = df_sents[high_conf_mask].copy()

# RQ3 aggregate metrics
agree = (df_sents["vader_label"] == df_sents["roberta_label"]).mean()
try:
    kappa_sent = cohen_kappa_score(
        df_sents["vader_label"], df_sents["roberta_label"],
        labels=["positive","neutral","negative"])
except Exception:
    kappa_sent = float("nan")

print(f"\n  RQ3 — VADER vs RoBERTa:")
print(f"  Overall agreement : {agree:.1%}")
print(f"  Cohen's Kappa     : {kappa_sent:.3f}")
print(f"\n  {'Label':<12} {'VADER':>8} {'RoBERTa':>9} {'Ensemble':>10}")
for lbl in ["positive","neutral","negative"]:
    v = (df_sents["vader_label"]   == lbl).sum()
    r = (df_sents["roberta_label"] == lbl).sum()
    e = (df_sents["sentiment"]     == lbl).sum()
    print(f"  {lbl:<12} {v:>8,} {r:>9,} {e:>10,}")

# Per-column agreement — columns with lower agreement indicate
# topics where sentiment is inherently more ambiguous
print(f"\n  Agreement by column:")
for col in COLUMNS_ORDER:
    sub = df_sents[df_sents["pred_col"]==col]
    if len(sub) == 0: continue
    agr = (sub["vader_label"]==sub["roberta_label"]).mean()*100
    print(f"  {col:<35} {agr:.1f}%")

df_sents.to_csv(str(OUTPUT_DIR / "classified_final.csv"), index=False)
df_sents_hc.to_csv(str(OUTPUT_DIR / "classified_final_highconf.csv"), index=False)
print(f"  ✅ Stage 6 complete")


# ════════════════════════════════════════════════════════════════
# STAGE 7 — OUTPUT GENERATION
# ════════════════════════════════════════════════════════════════
# Goal: Convert pipeline findings into interpretable outputs for
#       naval architects and academic reporting.
#
# Output files:
#   output_matrix.xlsx             — 6×4 KPI matrix (full dataset)
#   output_matrix_highconf.xlsx    — 6×4 KPI matrix (high-confidence subset)
#   01_sentiment_distribution.png  — Sentiment by column (RQ4; thesis Figure 10)
#   02_ship_heatmap.png            — Ship × column negativity heatmap (RQ4; thesis Figure 11)
#   03_row_distribution.png        — Internal response distribution (thesis Figure 5)
#   04_model_agreement.png         — VADER–RoBERTa agreement per column (RQ3; thesis Figure 8)
#   05_classifier_accuracy.png     — Per-class classifier accuracy (RQ2; thesis Figure 7)
#   06_baseline_comparison.png     — Ablation study visualisation (RQ2; thesis Figure 6)
#   07_confidence_analysis.png     — Confidence distribution + full vs high-conf KPI (thesis Figure 12)
#
# Excel output structure:
#   Rows: 4 internal response types (Pizam & Tasci 2019)
#   Columns: 6 environmental factors (Akter & Romanoff, 2024; the sixth
#            column, Navigation & Exterior, is added in this thesis)
#   Each cell: net sentiment score (▲positive − ▼negative)
#   Cell colour: green = net positive, red = net negative
#   This is an automated, scaled version of the matrix originally
#   constructed manually by Akter & Romanoff (2024) on 172 reviews.
# ════════════════════════════════════════════════════════════════
print("\n" + "━"*62)
print("  STAGE 7 — Output generation")
print("━"*62)

df_c = df_sents.copy()
ship_counts  = df_c["ship_name"].value_counts()
# Ships below MIN_REVIEWS_FOR_SHIP are excluded from ship-level sheets
insufficient = ship_counts[ship_counts < MIN_REVIEWS_FOR_SHIP].index.tolist()
df_suf       = df_c[~df_c["ship_name"].isin(insufficient)]

# Short column labels for Excel headers (fit within cell width)
COL_LABELS = {
    "Layout / Design Factors":    "Layout /\nDesign Factors",
    "Ambient Factors":            "Ambient\nFactors",
    "Social Factors":             "Social\nFactors",
    "Product / Service Factors":  "Product /\nService Factors",
    "On-board Enjoyment Factors": "On-board\nEnjoyment Factors",
    "Navigation & Exterior":      "Navigation &\nExterior",
}

def build_output_sheet(ws, data_df, title):
    """
    Builds one sheet of the output Excel workbook for a given dataset slice.
    Each cell shows: net score (top) and positive/negative counts (bottom).
    Green = net positive; red = net negative.
    A methodology footer is appended for reproducibility.
    """
    n = len(COLUMNS_ORDER)

    # Merged title row
    ws.merge_cells(f"A1:{chr(65+n+1)}1")
    xl_header_cell(ws, 1, 1,
        f"On-board Environmental Factors — {title} | "
        f"Akter & Romanoff (2024) | "
        f"Col:{col_acc:.0%}(κ={col_kappa:.3f}) "
        f"Row:{row_acc:.0%}(κ={row_kappa:.3f}) | "
        f"Özen, Aalto 2026",
        fill_color=NAVY, size=9)
    ws.row_dimensions[1].height = 28

    # Column header row (row 3)
    xl_header_cell(ws, 3, 1, "Passenger\nInternal\nResponse")
    for ci, col in enumerate(COLUMNS_ORDER):
        xl_header_cell(ws, 3, ci+2, COL_LABELS[col])
    xl_header_cell(ws, 3, n+2, "Sum")
    xl_header_cell(ws, 3, n+3, "Alignment")
    ws.row_dimensions[3].height = 50

    cp = {col: 0 for col in COLUMNS_ORDER}   # cumulative positives (for totals row)
    cn = {col: 0 for col in COLUMNS_ORDER}   # cumulative negatives

    # Fill one row per internal response type
    for ri, resp_row in enumerate(ROWS_ORDER):
        r      = ri + 4
        row_df = data_df[data_df["pred_row"] == resp_row]
        ws.row_dimensions[r].height = 50
        xl_header_cell(ws, r, 1, resp_row, fill_color=DGRAY)
        rp = rn = 0   # row-level totals
        for ci, col in enumerate(COLUMNS_ORDER):
            cdf = row_df[row_df["pred_col"] == col]
            p   = int((cdf["sentiment"] == "positive").sum())
            nv  = int((cdf["sentiment"] == "negative").sum())
            net = p - nv
            cp[col] += p; cn[col] += nv; rp += p; rn += nv
            # Green for net positive, red for net negative
            xl_data_cell(ws, r, ci+2,
                f"{'+' if net>=0 else ''}{net}\n{p}▲/{nv}▼",
                fill_color=GREEN if net>=0 else RED,
                text_color="2D7A2D" if net>=0 else "AE1E1E", bold=True)
        net_r = rp - rn
        xl_header_cell(ws, r, n+2,
            f"{'+' if net_r>=0 else ''}{net_r}\n{rp}▲/{rn}▼", fill_color=DARK)
        # Alignment ratio: >1 indicates net positive experience
        xl_header_cell(ws, r, n+3,
            f"{rp/rn:.2f}×" if rn > 0 else "N/A", fill_color=DGRAY)

    # Column totals row (sum across all row types)
    sr = len(ROWS_ORDER) + 4
    ws.row_dimensions[sr].height = 50
    xl_header_cell(ws, sr, 1, "Sum", fill_color=DARK)
    tp = tn = 0
    for ci, col in enumerate(COLUMNS_ORDER):
        p = cp[col]; nv = cn[col]; net = p-nv; tp += p; tn += nv
        xl_data_cell(ws, sr, ci+2,
            f"{'+' if net>=0 else ''}{net}\n{p}▲/{nv}▼",
            fill_color=LBLUE, text_color=NAVY, bold=True)
    xl_header_cell(ws, sr, n+2, f"+{tp-tn}\n{tp}▲/{tn}▼", fill_color=DARK)
    xl_header_cell(ws, sr, n+3,
        f"{tp/tn:.2f}×" if tn > 0 else "N/A", fill_color=DGRAY)

    # Methodology footer — enables reproducibility review
    nr = sr + 2
    ws.cell(row=nr, column=1).value = (
        f"NEXUS v1.0 | Logistic Regression (TF-IDF n-gram(1–3) + MiniLM-L6-v2 embeddings) | "
        f"Trained on {len(labeled):,} Claude Haiku-annotated sentences | "
        f"Column κ={col_kappa:.3f} CV={col_cv.mean():.1%} | "
        f"Row κ={row_kappa:.3f} CV={row_cv.mean():.1%} | "
        f"Sentiment: VADER + RoBERTa ensemble κ={kappa_sent:.3f} | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ws.cell(row=nr, column=1).font = Font(italic=True, size=8, color="888888")
    ws.merge_cells(start_row=nr, start_column=1, end_row=nr, end_column=n+3)

    # Column widths
    ws.column_dimensions["A"].width = 16
    for ci in range(n):
        ws.column_dimensions[get_column_letter(ci+2)].width = 18
    ws.column_dimensions[get_column_letter(n+2)].width = 18
    ws.column_dimensions[get_column_letter(n+3)].width = 13

# Generate two Excel files: full dataset and high-confidence subset
for df_use, suffix in [(df_c, ""), (df_sents_hc, "_highconf")]:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    build_output_sheet(wb.create_sheet("All Ships"), df_use, "All Ships")
    df_suf_use = df_use[~df_use["ship_name"].isin(insufficient)]
    emojis = ["🚢","⛴️","🛳️","🚤","🛥️"]
    for i, ship in enumerate(sorted(df_suf_use["ship_name"].unique())):
        ws_s = wb.create_sheet(f"{emojis[i%5]} {ship[:25]}")
        build_output_sheet(ws_s, df_use[df_use["ship_name"]==ship], ship)
    xl_path = str(OUTPUT_DIR / f"output_matrix{suffix}.xlsx")
    wb.save(xl_path)
    print(f"  ✅ output_matrix{suffix}.xlsx")

# ── CHARTS ───────────────────────────────────────────────────────
def save_fig(name):
    plt.savefig(str(OUTPUT_DIR/name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {name}")

# Chart 1: Sentiment distribution by environmental factor (primary RQ4 figure — thesis Figure 10)
cat_stats = []
for col in COLUMNS_ORDER:
    sub = df_c[df_c["pred_col"]==col]
    if len(sub)==0: continue
    t = len(sub)
    cat_stats.append({"col":col,
        "neg":(sub["sentiment"]=="negative").sum()/t*100,
        "neu":(sub["sentiment"]=="neutral").sum()/t*100,
        "pos":(sub["sentiment"]=="positive").sum()/t*100,
        "total":t})
stats_df = pd.DataFrame(cat_stats).sort_values("neg", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,7))
fig.suptitle(
    f"Design-Response Matrix — Passenger Sentiment by Environmental Factor\n"
    f"(n={len(df_c):,} sentences | taxonomy: Akter & Romanoff, 2024 | "
    f"Col acc:{col_acc:.0%} Row acc:{row_acc:.0%})",
    fontsize=13, fontweight="bold", y=1.01)
y = np.arange(len(stats_df)); h = 0.6
ax1.barh(y, stats_df["neg"], h, color="#E74C3C", label="Negative")
ax1.barh(y, stats_df["neu"], h, left=stats_df["neg"], color="#BDC3C7", label="Neutral")
ax1.barh(y, stats_df["pos"], h, left=stats_df["neg"]+stats_df["neu"],
         color="#2ECC71", label="Positive")
for i, row in enumerate(stats_df.itertuples()):
    ax1.text(row.neg/2, i, f"{row.neg:.0f}%",
             ha="center", va="center", fontsize=9, fontweight="bold", color="white")
ax1.set_yticks(y); ax1.set_yticklabels(stats_df["col"], fontsize=10)
ax1.set_xlabel("Percentage (%)"); ax1.set_title("Sentiment Distribution", fontweight="bold")
ax1.set_xlim(0,100); ax1.legend(loc="lower right"); ax1.grid(axis="x", alpha=0.3)
colors_b = ["#E74C3C" if r.neg>=30 else "#F39C12" if r.neg>=22 else "#2ECC71"
            for r in stats_df.itertuples()]
bars = ax2.barh(y, stats_df["total"], h, color=colors_b)
for bar, row in zip(bars, stats_df.itertuples()):
    ax2.text(bar.get_width()+stats_df["total"].max()*0.01,
             bar.get_y()+bar.get_height()/2,
             f"{int(row.total):,}", va="center", fontsize=9)
ax2.set_yticks(y); ax2.set_yticklabels(stats_df["col"], fontsize=10)
ax2.set_xlabel("Number of Mentions"); ax2.set_title("Total Mentions per Column", fontweight="bold")
ax2.grid(axis="x", alpha=0.3)
plt.tight_layout(); save_fig("01_sentiment_distribution.png")

# Chart 2: Ship × column negativity heatmap (ship comparison, RQ4 — thesis Figure 11)
sorted_ships = sorted(df_suf["ship_name"].unique())
hdata = np.array([
    [(df_suf[(df_suf["pred_col"]==col)&(df_suf["ship_name"]==ship)]["sentiment"]=="negative").sum() /
     max(len(df_suf[(df_suf["pred_col"]==col)&(df_suf["ship_name"]==ship)]),1)*100
     for ship in sorted_ships] for col in COLUMNS_ORDER])
fig, ax = plt.subplots(figsize=(13,7))
im = ax.imshow(hdata, cmap="RdYlGn_r", aspect="auto", vmin=10, vmax=50)
ax.set_xticks(range(len(sorted_ships)))
ax.set_xticklabels(sorted_ships, rotation=30, ha="right")
ax.set_yticks(range(len(COLUMNS_ORDER))); ax.set_yticklabels(COLUMNS_ORDER)
for i in range(len(COLUMNS_ORDER)):
    for j in range(len(sorted_ships)):
        ax.text(j, i, f"{hdata[i,j]:.0f}%", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if hdata[i,j]>35 else "black")
plt.colorbar(im, ax=ax, label="Negative Sentiment %")
ax.set_title("Negative Sentiment % by Ship and Environmental Factor",
             fontsize=13, fontweight="bold")
plt.tight_layout(); save_fig("02_ship_heatmap.png")

# Chart 3: Internal response distribution (Pizam & Tasci 2019 framework)
row_counts = df_c["pred_row"].value_counts()
fig, ax = plt.subplots(figsize=(9,5))
colors_r = ["#3498DB","#E74C3C","#2ECC71","#F39C12"]
bars = ax.bar(ROWS_ORDER, [row_counts.get(r,0) for r in ROWS_ORDER], color=colors_r)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
            f"{int(bar.get_height()):,}", ha="center", fontsize=10)
ax.set_title("Passenger Internal Response Distribution (Pizam & Tasci 2019)",
             fontweight="bold")
ax.set_ylabel("Number of Sentences"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); save_fig("03_row_distribution.png")

# Chart 4: RQ3 — VADER vs RoBERTa agreement by column (thesis Figure 8)
agree_data = []
for col in COLUMNS_ORDER:
    sub = df_c[df_c["pred_col"]==col]
    if len(sub)==0: continue
    agree_data.append({"col":col,
        "agree":(sub["vader_label"]==sub["roberta_label"]).mean()*100})
ag_df = pd.DataFrame(agree_data)
fig, ax = plt.subplots(figsize=(11,5))
bc = ["#2ECC71" if v>=70 else "#F39C12" if v>=60 else "#E74C3C"
      for v in ag_df["agree"]]
bars = ax.bar(range(len(ag_df)), ag_df["agree"], color=bc)
ax.axhline(y=ag_df["agree"].mean(), color="gray", linestyle="--",
           label=f"Mean: {ag_df['agree'].mean():.1f}%")
for bar, val in zip(bars, ag_df["agree"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(ag_df)))
ax.set_xticklabels(ag_df["col"], rotation=20, ha="right")
ax.set_ylabel("Agreement (%)"); ax.set_ylim(0,105)
ax.set_title("VADER vs RoBERTa Agreement by Column",
             fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); save_fig("04_model_agreement.png")

# Chart 5: Per-class classifier accuracy (methodology validation figure)
test_results = pd.read_csv(str(OUTPUT_DIR/"test_set_results.csv"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
col_accs = [(test_results[test_results["claude_col"]==col]["claude_col"] ==
             test_results[test_results["claude_col"]==col]["col_pred"]).mean()*100
            if (test_results["claude_col"]==col).any() else 0
            for col in COLUMNS_ORDER]
bc2 = ["#2ECC71" if a>=70 else "#F39C12" if a>=50 else "#E74C3C" for a in col_accs]
bars1 = ax1.bar(range(len(COLUMNS_ORDER)), col_accs, color=bc2)
for bar, val in zip(bars1, col_accs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.0f}%", ha="center", fontsize=9)
ax1.set_xticks(range(len(COLUMNS_ORDER)))
ax1.set_xticklabels([c.split("/")[0].strip() for c in COLUMNS_ORDER],
                    rotation=20, ha="right")
ax1.set_ylim(0,105); ax1.grid(axis="y", alpha=0.3)
ax1.set_title(
    f"Column Classifier Accuracy\n"
    f"(overall: {col_acc:.0%}  κ={col_kappa:.3f}  CV={col_cv.mean():.0%}±{col_cv.std():.0%})",
    fontweight="bold")
ax1.axhline(y=col_acc*100, color="gray", linestyle="--", label="Overall mean")
ax1.legend()
row_accs = [(test_results[test_results["claude_row"]==row]["claude_row"] ==
             test_results[test_results["claude_row"]==row]["row_pred"]).mean()*100
            if (test_results["claude_row"]==row).any() else 0
            for row in ROWS_ORDER]
bc3 = ["#2ECC71" if a>=70 else "#F39C12" if a>=50 else "#E74C3C" for a in row_accs]
bars2 = ax2.bar(ROWS_ORDER, row_accs, color=bc3)
for bar, val in zip(bars2, row_accs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.0f}%", ha="center", fontsize=10)
ax2.set_ylim(0,105); ax2.grid(axis="y", alpha=0.3)
ax2.set_title(
    f"Row Classifier Accuracy\n"
    f"(overall: {row_acc:.0%}  κ={row_kappa:.3f}  CV={row_cv.mean():.0%}±{row_cv.std():.0%})",
    fontweight="bold")
ax2.axhline(y=row_acc*100, color="gray", linestyle="--", label="Overall mean")
ax2.legend()
plt.tight_layout(); save_fig("05_classifier_accuracy.png")

# Chart 6: Ablation study — contribution of each feature set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
names      = ["TF-IDF\nonly", "Embeddings\nonly", "TF-IDF +\nEmbeddings"]
col_accs_b = [r["col_acc"]*100 for r in baseline_results]
row_accs_b = [r["row_acc"]*100 for r in baseline_results]
colors_bl  = ["#95A5A6","#3498DB","#2ECC71"]   # grey, blue, green (increasing)

bars1 = ax1.bar(names, col_accs_b, color=colors_bl)
for bar, val in zip(bars1, col_accs_b):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylim(0,100); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Column Classifier\nBaseline Comparison", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

bars2 = ax2.bar(names, row_accs_b, color=colors_bl)
for bar, val in zip(bars2, row_accs_b):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylim(0,100); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Row Classifier\nBaseline Comparison", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Feature Ablation Study — Contribution of Each Feature Set",
             fontsize=13, fontweight="bold")
plt.tight_layout(); save_fig("06_baseline_comparison.png")

# Chart 7: Confidence distribution and full vs high-confidence KPI comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Column classifier confidence histogram
axes[0].hist(col_confs, bins=30, color="#3498DB", edgecolor="white", alpha=0.8)
axes[0].axvline(x=CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                label=f"Threshold = {CONFIDENCE_THRESHOLD}")
axes[0].set_xlabel("Confidence Score"); axes[0].set_ylabel("Number of Sentences")
axes[0].set_title("Column Classifier\nConfidence Distribution", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Row classifier confidence histogram
axes[1].hist(row_confs, bins=30, color="#E74C3C", edgecolor="white", alpha=0.8)
axes[1].axvline(x=CONFIDENCE_THRESHOLD, color="navy", linestyle="--",
                label=f"Threshold = {CONFIDENCE_THRESHOLD}")
axes[1].set_xlabel("Confidence Score"); axes[1].set_ylabel("Number of Sentences")
axes[1].set_title("Row Classifier\nConfidence Distribution", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

# Full dataset vs high-confidence subset: do KPI findings change?
neg_full = []
neg_hc   = []
for col in COLUMNS_ORDER:
    sub_f = df_c[df_c["pred_col"]==col]
    sub_h = df_sents_hc[df_sents_hc["pred_col"]==col]
    neg_full.append((sub_f["sentiment"]=="negative").mean()*100 if len(sub_f)>0 else 0)
    neg_hc.append(  (sub_h["sentiment"]=="negative").mean()*100 if len(sub_h)>0 else 0)

x = np.arange(len(COLUMNS_ORDER)); w = 0.35
axes[2].bar(x-w/2, neg_full, w, label="Full dataset", color="#3498DB", alpha=0.8)
axes[2].bar(x+w/2, neg_hc,   w, label="High-confidence only", color="#E74C3C", alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels([c.split("/")[0].strip() for c in COLUMNS_ORDER],
                        rotation=20, ha="right", fontsize=9)
axes[2].set_ylabel("Negative Sentiment (%)"); axes[2].set_ylim(0,70)
axes[2].set_title("Negative Sentiment %:\nFull Dataset vs High-Confidence Subset",
                  fontweight="bold")
axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout(); save_fig("07_confidence_analysis.png")

# ── SUMMARY REPORT ───────────────────────────────────────────────
total = len(df_c)
pos_n = (df_c["sentiment"]=="positive").sum()
neg_n = (df_c["sentiment"]=="negative").sum()
neu_n = (df_c["sentiment"]=="neutral").sum()

human_section = ""
if ch_col_kappa is not None:
    human_section = f"""
HUMAN ANNOTATION VALIDATION
  Claude vs Human — Column Kappa : {ch_col_kappa:.3f}
  Claude vs Human — Row Kappa    : {ch_row_kappa:.3f}
  Number of human-labeled sentences : {N_HUMAN_ANNOTATE}
"""

report = f"""
{'='*65}
  NEXUS PIPELINE v1.0 — SUMMARY REPORT
  {datetime.now().strftime('%d %B %Y, %H:%M')}
  Taxonomy: Akter & Romanoff (2024), extended to 6x4 in this thesis
  Author: Ulas Özen, Aalto University, 2026
{'='*65}

METHODOLOGY
  Classification : Domain-adapted Logistic Regression
                   Features: TF-IDF (n-gram 1–3) + MiniLM-L6-v2 embeddings
  Training data  : {len(labeled):,} sentences annotated by Claude Haiku
  Annotation     : Claude Haiku (claude-haiku-4-5-20251001)
  Sentiment      : VADER + RoBERTa-base ensemble
  Validation     : {CV_FOLDS}-fold stratified CV + held-out test set ({int(TEST_SIZE*100)}%)

CLASSIFIER PERFORMANCE
  Column: {col_acc:.1%} accuracy  κ={col_kappa:.3f}  CV={col_cv.mean():.1%} ± {col_cv.std():.1%}
  Row   : {row_acc:.1%} accuracy  κ={row_kappa:.3f}  CV={row_cv.mean():.1%} ± {row_cv.std():.1%}

ABLATION STUDY (feature contribution)
  {'Model':<30} {'Col Acc':>8} {'Col κ':>7} {'Row Acc':>8} {'Row κ':>7}
  {'-'*62}
{chr(10).join(f"  {r['name']:<30} {r['col_acc']:>7.1%} {r['col_kappa']:>7.3f} {r['row_acc']:>8.1%} {r['row_kappa']:>7.3f}" for r in baseline_results)}

STATISTICAL SIGNIFICANCE (McNemar Test — Full vs TF-IDF only)
  Column: p={p_col:.4f}  {'✅ significant' if p_col<0.05 else '⚠️ not significant'}
  Row   : p={p_row:.4f}  {'✅ significant' if p_row<0.05 else '⚠️ not significant'}
{human_section}
CONFIDENCE ANALYSIS
  High confidence (≥{CONFIDENCE_THRESHOLD}) : {n_high:,} sentences ({n_high/len(df_sents)*100:.1f}%)
  Low confidence          : {(~high_conf_mask).sum():,} sentences ({(~high_conf_mask).mean()*100:.1f}%)
  (Both full and high-confidence Excel outputs are saved)

DATASET
  Reviews              : {len(df):,}
  Classified sentences : {total:,}
  Positive             : {pos_n:,} ({pos_n/total*100:.1f}%)
  Neutral              : {neu_n:,} ({neu_n/total*100:.1f}%)
  Negative             : {neg_n:,} ({neg_n/total*100:.1f}%)
  Positive / Negative  : {pos_n/neg_n:.2f}×

KEY FINDINGS — ENVIRONMENTAL FACTOR PRIORITIES
{chr(10).join(f"  {col:<35} {(df_c[df_c['pred_col']==col]['sentiment']=='negative').mean()*100:.1f}%" + (" 🔴 HIGH PRIORITY" if (df_c[df_c['pred_col']==col]['sentiment']=='negative').mean()*100>=30 else " 🟡 MODERATE" if (df_c[df_c['pred_col']==col]['sentiment']=='negative').mean()*100>=22 else " 🟢 LOW") for col in sorted(COLUMNS_ORDER, key=lambda c:-(df_c[df_c['pred_col']==c]['sentiment']=='negative').mean()))}

SHIP COMPARISON (fleet average: {neg_n/total*100:.1f}% negative)
{chr(10).join(f"  {ship:<35} {(df_c[df_c['ship_name']==ship]['sentiment']=='negative').mean()*100:.1f}%  ({(df_c[df_c['ship_name']==ship]['sentiment']=='negative').mean()*100-neg_n/total*100:+.1f}% vs fleet)" + (" ⚠️ above average" if (df_c[df_c['ship_name']==ship]['sentiment']=='negative').mean()*100-neg_n/total*100>3 else "") for ship in sorted(df_suf["ship_name"].unique()))}

RQ3 — SENTIMENT MODEL COMPARISON
  VADER–RoBERTa overall agreement : {agree:.1%}
  Cohen's Kappa                   : {kappa_sent:.3f}

OUTPUT FILES
  output_matrix.xlsx             — Full dataset KPI matrix
  output_matrix_highconf.xlsx    — High-confidence subset KPI matrix
  01–07 PNG charts               — See filenames for details
  human_annotation_task.csv      — Sentences for human ground-truth labeling
  test_set_results.csv           — Held-out test set with predictions (error analysis)
  labeled_dataset.csv            — Claude-annotated training data
  annotation_cache.csv           — Annotation cache (resume support)

{'='*65}
  ✅ NEXUS Pipeline v1.0 complete
{'='*65}
"""

print(report)
with open(str(OUTPUT_DIR/"summary_report.txt"), "w") as f:
    f.write(report)

print(f"\n  📁 All outputs saved to: {OUTPUT_DIR}")
print(f"  🎯 NEXUS Pipeline v1.0 complete!")
