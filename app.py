import streamlit as st
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig, RobertaModel
import sys
from types import SimpleNamespace
import json
import urllib.request
import os

# Page config
st.set_page_config(
    page_title="VulBERT - Code Vulnerability Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
    <style>
        :root {
            --primary-color: #00D9FF;
            --secondary-color: #FF006E;
            --background-color: #0a0e27;
            --surface-color: #1a1f3a;
            --text-color: #e0e0e0;
            --success-color: #00ff88;
            --warning-color: #ffaa00;
            --danger-color: #ff3366;
        }
        
        * {
            font-family: 'Courier New', monospace;
        }
        
        body {
            background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
            color: var(--text-color);
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
        }
        
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(255, 0, 110, 0.1));
            border-radius: 15px;
            border: 2px solid rgba(0, 217, 255, 0.3);
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: rgba(255, 0, 110, 0.1);
            border: 2px solid rgba(255, 0, 110, 0.3);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 0 20px rgba(255, 0, 110, 0.2);
        }
        
        .safe-card {
            background: rgba(0, 255, 136, 0.1);
            border: 2px solid rgba(0, 255, 136, 0.3);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        }
        
        .vulnerable-line {
            background: rgba(255, 0, 110, 0.2);
            border-left: 4px solid rgba(255, 0, 110, 0.8);
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            font-size: 0.95rem;
        }
        
        .safe-line {
            background: rgba(0, 255, 136, 0.05);
            border-left: 4px solid rgba(0, 255, 136, 0.5);
            padding: 0.5rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            font-size: 0.95rem;
        }
        
        .code-block {
            background: rgba(10, 14, 39, 0.8);
            border: 2px solid rgba(0, 217, 255, 0.3);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            overflow-x: auto;
        }
        
        .line-number {
            color: rgba(0, 217, 255, 0.6);
            margin-right: 1rem;
            font-weight: bold;
            min-width: 3rem;
            text-align: right;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #00D9FF, #00a8cc);
            color: #0a0e27;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            font-size: 1rem;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 217, 255, 0.5);
        }
        
        .stTextArea, .stNumberInput, .stSelectbox {
            background: rgba(26, 31, 58, 0.5) !important;
            border: 2px solid rgba(0, 217, 255, 0.3) !important;
            border-radius: 8px !important;
            color: var(--text-color) !important;
        }
        
        .prediction-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2rem;
            margin: 0.5rem 0;
        }
        
        .safe-badge {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.3), rgba(0, 255, 136, 0.1));
            border: 2px solid rgba(0, 255, 136, 0.6);
            color: #00ff88;
        }
        
        .vulnerable-badge {
            background: linear-gradient(135deg, rgba(255, 0, 110, 0.3), rgba(255, 0, 110, 0.1));
            border: 2px solid rgba(255, 0, 110, 0.6);
            color: #ff3366;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(0, 217, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transition: width 0.3s ease;
        }
        
        h1, h2, h3 {
            color: var(--primary-color);
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
        }
        
        .sidebar-title {
            color: var(--primary-color);
            font-size: 1.3rem;
            font-weight: bold;
            margin: 1rem 0 0.5rem 0;
        }
        
        .info-box {
            background: rgba(0, 217, 255, 0.1);
            border: 2px solid rgba(0, 217, 255, 0.3);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'coarse_model' not in st.session_state:
    st.session_state.coarse_model = None
if 'line_model' not in st.session_state:
    st.session_state.line_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cpu')

def has_hf_weights(path: Path) -> bool:
    """Check if a path contains HuggingFace model weights."""
    return (path / 'pytorch_model.bin').exists() or (path / 'model.safetensors').exists()

def download_model_from_release(model_path: Path, release_url: str, filename: str):
    """Download model file from GitHub release if it doesn't exist."""
    if model_path.exists():
        print(f"✅ {filename} already exists, skipping download")
        return True
    
    try:
        print(f"⬇️ Downloading {filename} from GitHub release...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(release_url, model_path)
        print(f"✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False




# @st.cache_resource
# def load_models():
#     """Load pre-trained VulBERTa models for both coarse and line-level detection."""
#     try:
#         PROJECT_ROOT = Path.cwd()
#         RESOURCE_DIR = PROJECT_ROOT / 'resource'
#         CODEBERT_DIR = RESOURCE_DIR / 'codebert-base'
        
#         # Priority order for coarse model: fine-tuned VulBERTa > checkpoint > base
#         vulberta_finetuned_dir = PROJECT_ROOT / 'new_model' / 'vulberta_finetuned'
#         coarse_checkpoint_dir = PROJECT_ROOT / 'new_model' / 'vul' / 'checkpoint-best-f1'
#         vulberta_base_dir = RESOURCE_DIR / 'VulBERTa'
        
#         device = torch.device('cpu')
        
#         # Step 1: Load tokenizer from CodeBERT (50265 vocab)
#         tokenizer = AutoTokenizer.from_pretrained(str(CODEBERT_DIR))
#         tokenizer_vocab_size = len(tokenizer)
        
#         # Step 2: Load coarse-grained model with priority order
#         coarse_model = None
#         coarse_model_path = None
        
#         # Try fine-tuned VulBERTa first
#         if vulberta_finetuned_dir.exists():
#             print("[VulBERT] Loading fine-tuned VulBERTa model...")
#             config = RobertaConfig.from_pretrained(str(vulberta_finetuned_dir))
#             coarse_model = RobertaForSequenceClassification.from_pretrained(str(vulberta_finetuned_dir), config=config)
#             coarse_model_path = vulberta_finetuned_dir
#             print("[VulBERT] Fine-tuned VulBERTa loaded successfully")
        
#         # Fallback to checkpoint
#         elif coarse_checkpoint_dir.exists():
#             print("[VulBERT] Loading coarse checkpoint model...")
#             config = RobertaConfig.from_pretrained(str(coarse_checkpoint_dir))
#             coarse_model = RobertaForSequenceClassification.from_pretrained(str(coarse_checkpoint_dir), config=config)
#             coarse_model_path = coarse_checkpoint_dir
#             print("[VulBERT] Coarse checkpoint loaded successfully")
        
#         # Last resort: convert base VulBERTa from MLM to classification
#         elif vulberta_base_dir.exists():
#             print("[VulBERT] Converting base VulBERTa from MLM to classification...")
#             config = RobertaConfig.from_pretrained(str(vulberta_base_dir))
#             config.num_labels = 2
#             config.architectures = ["RobertaForSequenceClassification"]
            
#             # Load base model and convert
#             base_model = RobertaModel.from_pretrained(str(vulberta_base_dir), config=config)
#             coarse_model = RobertaForSequenceClassification(config=config)
#             coarse_model.roberta.load_state_dict(base_model.state_dict(), strict=False)
#             coarse_model_path = vulberta_base_dir
#             print("[VulBERT] Base VulBERTa converted to classification")
        
#         else:
#             raise FileNotFoundError("No VulBERTa model found in any expected location")
        
#         # Resize embeddings if needed
#         if coarse_model is not None:
#             model_vocab_size = coarse_model.config.vocab_size
#             if model_vocab_size != tokenizer_vocab_size:
#                 print(f"[VulBERT] Vocab mismatch: Model {model_vocab_size}, tokenizer {tokenizer_vocab_size}")
#                 print(f"[VulBERT] Resizing embeddings...")
#                 coarse_model.resize_token_embeddings(tokenizer_vocab_size)
#                 print(f"[VulBERT] Model resized successfully")
            
#             coarse_model.to(device)
#             coarse_model.eval()
        
#         # Step 3: Load line-level model from staged-models
#         from Models.linevul_model import Model as LineVulEncoder
#         from types import SimpleNamespace
        
#         line_config = RobertaConfig.from_pretrained(str(CODEBERT_DIR))
#         line_config.num_labels = 2
#         base_encoder = RobertaForSequenceClassification.from_pretrained(str(CODEBERT_DIR), config=line_config)
        
#         # Create line-level model wrapper
#         args = SimpleNamespace(device=device)
#         line_model = LineVulEncoder(base_encoder, line_config, tokenizer, args)
        
#         # Load pre-trained weights if available
#         STAGED_MODELS_DIR = PROJECT_ROOT / 'new_model' / 'line_vul' / 'checkpoint-best-f1'
#         model_path = STAGED_MODELS_DIR / 'model_2048.bin'


#         if model_path.exists():
#             print(f"[VulBERT] Loading line-level model from {model_path}")
#             state_dict = torch.load(str(model_path), map_location=device)
#             line_model.load_state_dict(state_dict, strict=False)
#             print(f"[VulBERT] Line-level model loaded successfully")
#         else:
#             print(f"[VulBERT] Warning: Line-level model not found at {model_path}, using untrained model")
        
#         line_model.to(device)
#         line_model.eval()
        
#         # Store metadata
#         coarse_model.model_name = f"VulBERTa (Coarse) - {coarse_model_path.name}"
#         coarse_model.training_data = "BigVul (188K functions)"
#         line_model.model_name = "LineVul (Fine-grained)"
#         line_model.training_data = "BigVul (line-level)"
        
#         return coarse_model, line_model, tokenizer, device
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None, None, None


@st.cache_resource
def load_models():
    """Load VulBERTa (coarse) and LineVul (fine-grained) models with automatic download from GitHub releases."""
    import torch
    from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig
    from Models.linevul_model import Model as LineVulEncoder
    from types import SimpleNamespace
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()
    device = torch.device("cpu")

    # GitHub release URLs (update these with your actual release URLs)
    GITHUB_REPO = "HackLoading/CVD-project-VII"
    RELEASE_TAG = "v1.0.0"  # Update this to your release tag
    
    base_release_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

    # --- Paths ---
    codebert_dir = PROJECT_ROOT / "resource" / "codebert-base"
    coarse_dir = PROJECT_ROOT / "new_model" / "vul" / "checkpoint-best-f1"
    coarse_model_path = coarse_dir / "model.safetensors"
    line_model_dir = PROJECT_ROOT / "new_model" / "line_vul" / "checkpoint-best-f1"
    line_model_path = line_model_dir / "model_2048.bin"

    print(f"[DEBUG] Coarse model path: {coarse_model_path}")
    print(f"[DEBUG] Line model path: {line_model_path}")

    # --- Download models if not present ---
    print("🔄 Checking for model files...")
    
    # Download coarse model
    coarse_release_url = f"{base_release_url}/model.safetensors"
    if not download_model_from_release(coarse_model_path, coarse_release_url, "coarse model (model.safetensors)"):
        st.error("❌ Failed to download coarse model. Please check your internet connection and try again.")
        return None, None, None, device
    
    # Download line model
    line_release_url = f"{base_release_url}/model_2048.bin"
    if not download_model_from_release(line_model_path, line_release_url, "line model (model_2048.bin)"):
        st.error("❌ Failed to download line model. Please check your internet connection and try again.")
        return None, None, None, device

    # --- Load Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(codebert_dir))
    except Exception as e:
        print(f"❌ Tokenizer failed to load: {e}")
        st.error(f"❌ Failed to load tokenizer: {e}")
        return None, None, None, device

    # --- Load Coarse Model (Function-level VulBERTa) ---
    try:
        coarse_config = RobertaConfig.from_pretrained(str(coarse_dir))
        coarse_model = RobertaForSequenceClassification.from_pretrained(
            str(coarse_dir), config=coarse_config
        )
        coarse_model.to(device).eval()
        print("✅ Coarse (function-level) model loaded.")
    except Exception as e:
        print(f"❌ Coarse model failed to load: {e}")
        st.error(f"❌ Failed to load coarse model: {e}")
        coarse_model = None

    # --- Load Line Model (LineVul) ---
    try:
        base_encoder = RobertaForSequenceClassification.from_pretrained(str(codebert_dir))
        args = SimpleNamespace(device=device)
        line_model = LineVulEncoder(base_encoder, coarse_config, tokenizer, args)

        state_dict = torch.load(str(line_model_path), map_location=device)
        missing, unexpected = line_model.load_state_dict(state_dict, strict=False)
        print(f"✅ Line-level model loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        line_model.to(device).eval()

    except Exception as e:
        print(f"❌ Line model failed to load: {e}")
        st.error(f"❌ Failed to load line model: {e}")
        line_model = None

    return coarse_model, line_model, tokenizer, device


@torch.no_grad()
def predict_coarse(code, model, tokenizer, device, max_len=512):
    """Predict function-level vulnerability."""
    model.eval()
    enc = tokenizer(code, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits if hasattr(out, 'logits') else out[0]
    
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    
    return {
        'prediction': 'Vulnerable' if pred == 1 else 'Safe',
        'label': pred,
        'prob_safe': float(probs[0]),
        'prob_vulnerable': float(probs[1])
    }

@torch.no_grad()
def predict_lines(code, model, tokenizer, device, max_len=512):
    """Predict vulnerable lines using hybrid approach: ML model + pattern detection."""
    model.eval()
    lines = code.split('\n')
    scores = []

    # Common vulnerable patterns (case-insensitive)
    vulnerable_patterns = [
        r'\bstrcpy\s*\(',           # strcpy calls
        r'\bgets\s*\(',             # gets calls
        r'\bsprintf\s*\(',          # sprintf calls
        r'\bscanf\s*\([^)]*%s',     # scanf with %s
        r'\bcin\s*>>',              # C++ cin without bounds
        r'\bfree\s*\([^)]+\)\s*;\s*[^/]*\*\s*\w+',  # free followed by pointer access
        r'\bdelete\s*\[\]',         # delete[] (potential double free)
        r'\bmalloc\s*\([^)]+\)\s*;\s*[^/]*\*\s*\w+\s*=.*\w+',  # malloc followed by unchecked access
    ]

    import re

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            scores.append({'line_no': idx, 'code': line, 'prob_vul': 0.0})
            continue

        # Start with ML model prediction
        try:
            enc = tokenizer(line, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
            ids = enc['input_ids'].to(device)
            attn = enc['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=ids)
                if isinstance(outputs, tuple):
                    probs = outputs[0]
                else:
                    probs = outputs
                prob_vul = float(probs[0, 1].item())
        except Exception as e:
            # Fallback if model fails
            prob_vul = 0.5

        # Boost score if line matches vulnerable patterns
        line_lower = line.lower().strip()
        pattern_boost = 0.0

        for pattern in vulnerable_patterns:
            if re.search(pattern, line_lower, re.IGNORECASE):
                pattern_boost = 0.3  # Significant boost for pattern matches
                break

        # Additional heuristics
        if 'overflow' in line_lower or 'buffer' in line_lower:
            pattern_boost = max(pattern_boost, 0.2)

        if 'free' in line_lower and ('*' in line or '->' in line):
            pattern_boost = max(pattern_boost, 0.25)  # Use-after-free pattern

        # Combine ML prediction with pattern boost
        final_prob = min(1.0, prob_vul * 0.8 + pattern_boost)


        # Ensure some lines with obvious patterns get high scores
        if final_prob > 0.4:
           risk_level = "🔴 HIGH"


        scores.append({'line_no': idx, 'code': line, 'prob_vul': final_prob})

    sorted_scores = sorted(scores, key=lambda x: x['prob_vul'], reverse=True)
    return {'all_lines': scores, 'sorted_lines': sorted_scores}

@torch.no_grad()
def validate_model(model, tokenizer, device):
    """Validate that model is properly trained for vulnerability detection."""
    
    # Test cases: (code, expected_label, description)
    test_cases = [
        ("strcpy(buffer, input);", 1, "Buffer overflow - should be VULNERABLE"),
        ("gets(str);", 1, "Unsafe input - should be VULNERABLE"),
        ("printf(\"hello\");", 0, "Safe printf - should be SAFE"),
        ("int x = 5;", 0, "Simple assignment - should be SAFE"),
    ]
    
    model.eval()
    results = {"correct": 0, "total": 0, "details": []}
    
    for code, expected_label, description in test_cases:
        enc = tokenizer(code, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits if hasattr(out, 'logits') else out[0]
        
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        
        is_correct = pred_label == expected_label
        results["correct"] += int(is_correct)
        results["total"] += 1
        results["details"].append({
            "code": code,
            "expected": expected_label,
            "predicted": pred_label,
            "confidence": confidence,
            "correct": is_correct,
            "description": description
        })
    
    # Model is considered valid if it gets at least 75% of test cases correct
    validation_pass = (results["correct"] / results["total"]) >= 0.75
    
    return {
        "valid": validation_pass,
        "accuracy": results["correct"] / results["total"],
        "score": results["correct"],
        "total": results["total"],
        "details": results["details"]
    }

@torch.no_grad()
def debug_line_model_predictions(model, tokenizer, device, test_lines=None):
    """Debug line-level model predictions with detailed output."""
    if test_lines is None:
        test_lines = [
            "strcpy(buffer, input);  // VULNERABLE",
            "char buffer[32];",
            "void vulnerable_function(char *input) {",
            "}",
            "printf(\"Hello World\");",
            "int x = 5;",
        ]
    
    print("🔍 Debugging Line-Level Model Predictions")
    print("=" * 60)
    
    model.eval()
    results = []
    
    for line in test_lines:
        print(f"\n📝 Testing line: {line}")
        
        # Tokenize
        enc = tokenizer(line, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        
        with torch.no_grad():
            # Get raw model output
            outputs = model(input_ids=ids)
            print(f"  Raw outputs type: {type(outputs)}")
            
            if isinstance(outputs, tuple):
                probs = outputs[0]
                print(f"  Tuple output - probs shape: {probs.shape}")
            else:
                probs = outputs
                print(f"  Direct output - probs shape: {probs.shape}")
            
            # Extract probabilities
            prob_safe = float(probs[0, 0].item())
            prob_vul = float(probs[0, 1].item())
            
            print(".4f")
            print(".4f")
            
            prediction = 'VULNERABLE' if prob_vul > prob_safe else 'SAFE'
            confidence = max(prob_safe, prob_vul)
            
            print(f"  Prediction: {prediction} (confidence: {confidence:.4f})")
            
            results.append({
                'line': line,
                'prob_safe': prob_safe,
                'prob_vul': prob_vul,
                'prediction': prediction,
                'confidence': confidence
            })
    
    print("\n📊 Summary:")
    print(f"  Total lines tested: {len(results)}")
    vul_count = sum(1 for r in results if r['prediction'] == 'VULNERABLE')
    print(f"  Predicted vulnerable: {vul_count}")
    print(f"  Predicted safe: {len(results) - vul_count}")
    
    return results

@torch.no_grad()
def validate_line_model(model, tokenizer, device):
    """Validate line-level model with known vulnerable and safe patterns."""
    
    # Test cases with expected vulnerability scores
    test_cases = [
        # High vulnerability expected
        ("strcpy(buffer, input);", "high", "Classic buffer overflow"),
        ("gets(user_input);", "high", "Unsafe input function"),
        ("sprintf(buf, \"%s\", data);", "high", "Unbounded sprintf"),
        
        # Medium vulnerability expected
        ("char buffer[100];", "medium", "Buffer declaration (context dependent)"),
        ("free(ptr); *ptr = 0;", "medium", "Potential use-after-free"),
        
        # Low vulnerability expected (safe)
        ("printf(\"Hello World\");", "low", "Safe printf"),
        ("int x = 5;", "low", "Simple assignment"),
        ("return 0;", "low", "Return statement"),
        ("if (x > 0) {", "low", "Control flow"),
    ]
    
    model.eval()
    results = {"high": [], "medium": [], "low": [], "summary": {}}
    
    print("🔬 Validating Line-Level Model")
    print("=" * 50)
    
    for code, expected_risk, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Code: {code}")
        
        # Get prediction
        enc = tokenizer(code, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=ids)
            if isinstance(outputs, tuple):
                probs = outputs[0]
            else:
                probs = outputs
            prob_vul = float(probs[0, 1].item())
        
        # Categorize actual risk
        if prob_vul > 0.7:
            actual_risk = "high"
        elif prob_vul > 0.3:
            actual_risk = "medium"
        else:
            actual_risk = "low"
        
        correct = (expected_risk == actual_risk)
        status = "✓" if correct else "✗"
        
        print(".4f")
        print(f"Expected: {expected_risk}, Got: {actual_risk} {status}")
        
        results[expected_risk].append({
            "code": code,
            "prob_vul": prob_vul,
            "expected": expected_risk,
            "actual": actual_risk,
            "correct": correct,
            "description": description
        })
    
    # Calculate summary statistics
    total_tests = len(test_cases)
    correct_predictions = sum(len(results[risk]) for risk in ["high", "medium", "low"] 
                            for item in results[risk] if item["correct"])
    
    results["summary"] = {
        "total_tests": total_tests,
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / total_tests,
        "high_risk_cases": len(results["high"]),
        "medium_risk_cases": len(results["medium"]),
        "low_risk_cases": len(results["low"])
    }
    
    print("\n📈 Validation Summary:")
    print(f"  Accuracy: {results['summary']['accuracy']:.1%} ({correct_predictions}/{total_tests})")
    print(f"  High risk cases: {results['summary']['high_risk_cases']}")
    print(f"  Medium risk cases: {results['summary']['medium_risk_cases']}")
    print(f"  Low risk cases: {results['summary']['low_risk_cases']}")
    
    return results

def load_models_standalone():
    """Load pre-trained VulBERTa models without streamlit dependencies."""
    try:
        PROJECT_ROOT = Path.cwd()
        RESOURCE_DIR = PROJECT_ROOT / 'resource'
        CODEBERT_DIR = RESOURCE_DIR / 'codebert-base'
        
        # Priority order for coarse model: fine-tuned VulBERTa > checkpoint > base
        vulberta_finetuned_dir = PROJECT_ROOT / 'new_model' / 'vulberta_finetuned'
        coarse_checkpoint_dir = PROJECT_ROOT / 'new_model' / 'vul' / 'checkpoint-best-f1'
        vulberta_base_dir = RESOURCE_DIR / 'VulBERTa'
        
        device = torch.device('cpu')
        
        # Step 1: Load tokenizer from CodeBERT (50265 vocab)
        tokenizer = AutoTokenizer.from_pretrained(str(CODEBERT_DIR))
        tokenizer_vocab_size = len(tokenizer)
        
        # Step 2: Load coarse-grained model with priority order
        coarse_model = None
        coarse_model_path = None
        
        # Try fine-tuned VulBERTa first
        if vulberta_finetuned_dir.exists() and has_hf_weights(vulberta_finetuned_dir):
            print("[VulBERT] Loading fine-tuned VulBERTa model...")
            config = RobertaConfig.from_pretrained(str(vulberta_finetuned_dir))
            coarse_model = RobertaForSequenceClassification.from_pretrained(str(vulberta_finetuned_dir), config=config)
            coarse_model_path = vulberta_finetuned_dir
            print("[VulBERT] Fine-tuned VulBERTa loaded successfully")
        
        # Fallback to checkpoint
        elif coarse_checkpoint_dir.exists() and has_hf_weights(coarse_checkpoint_dir):
            print("[VulBERT] Loading coarse checkpoint model...")
            config = RobertaConfig.from_pretrained(str(coarse_checkpoint_dir))
            coarse_model = RobertaForSequenceClassification.from_pretrained(str(coarse_checkpoint_dir), config=config)
            coarse_model_path = coarse_checkpoint_dir
            print("[VulBERT] Coarse checkpoint loaded successfully")
        
        # Last resort: convert base VulBERTa from MLM to classification
        elif vulberta_base_dir.exists():
            print("[VulBERT] Converting base VulBERTa from MLM to classification...")
            config = RobertaConfig.from_pretrained(str(vulberta_base_dir))
            config.num_labels = 2
            config.architectures = ["RobertaForSequenceClassification"]
            
            # Load base model and convert
            base_model = RobertaModel.from_pretrained(str(vulberta_base_dir), config=config)
            coarse_model = RobertaForSequenceClassification(config=config)
            coarse_model.roberta.load_state_dict(base_model.state_dict(), strict=False)
            coarse_model_path = vulberta_base_dir
            print("[VulBERT] Base VulBERTa converted to classification")
        
        else:
            raise FileNotFoundError("No VulBERTa model found in any expected location")
        
        # Resize embeddings if needed
        if coarse_model is not None:
            model_vocab_size = coarse_model.config.vocab_size
            if model_vocab_size != tokenizer_vocab_size:
                print(f"[VulBERT] Vocab mismatch: Model {model_vocab_size}, tokenizer {tokenizer_vocab_size}")
                print(f"[VulBERT] Resizing embeddings...")
                coarse_model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"[VulBERT] Model resized successfully")
            
            coarse_model.to(device)
            coarse_model.eval()
        
        # Step 3: Load line-level model from staged-models
        from Models.linevul_model import Model as LineVulEncoder
        from types import SimpleNamespace
        
        line_config = RobertaConfig.from_pretrained(str(CODEBERT_DIR))
        line_config.num_labels = 2
        base_encoder = RobertaForSequenceClassification.from_pretrained(str(CODEBERT_DIR), config=line_config)
        
        # Create line-level model wrapper
        args = SimpleNamespace(device=device)
        line_model = LineVulEncoder(base_encoder, line_config, tokenizer, args)
        
        # Load pre-trained weights if available
        STAGED_MODELS_DIR = PROJECT_ROOT / 'new_model' / 'line_vul' / 'checkpoint-best-f1'
        model_path = STAGED_MODELS_DIR / 'model_2048.bin'

        if model_path.exists():
            print(f"[VulBERT] Loading line-level model from {model_path}")
            state_dict = torch.load(str(model_path), map_location=device)
            line_model.load_state_dict(state_dict, strict=False)
            print(f"[VulBERT] Line-level model loaded successfully")
        else:
            print(f"[VulBERT] Warning: Line-level model not found at {model_path}, using untrained model")
        
        line_model.to(device)
        line_model.eval()
        
        # Store metadata
        coarse_model.model_name = f"VulBERTa (Coarse) - {coarse_model_path.name}"
        coarse_model.training_data = "BigVul (188K functions)"
        line_model.model_name = "LineVul (Fine-grained)"
        line_model.training_data = "BigVul (line-level)"
        
        return coarse_model, line_model, tokenizer, device
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def render_code_with_highlighting(code, line_scores, threshold=0.45):
    """Render code with vulnerability highlighting."""
    lines = code.split('\n')
    
    for i, line in enumerate(lines, start=1):
        score = next((s['prob_vul'] for s in line_scores if s['line_no'] == i), 0.0)
        
        if score > threshold:
            st.markdown(f"""
                <div class="vulnerable-line">
                    <span class="line-number">{i:3d}</span>
                    <span style="color: #ff3366; font-weight: bold;">⚠️</span>
                    <code>{line}</code>
                    <span style="float: right; color: #ff3366; font-size: 0.85rem;">Risk: {score:.1%}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="safe-line">
                    <span class="line-number">{i:3d}</span>
                    <code>{line}</code>
                </div>
            """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🔍 VulBERT - Code Vulnerability Detector</h1>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            AI-Powered Security Analysis for Code Vulnerabilities
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    threshold = st.slider(
        "Vulnerability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Lines with risk score above this threshold will be highlighted"
    )
    
    show_all_lines = st.checkbox("Show All Lines", value=False)
    
    st.markdown("---")
    
    # Model Information Section
    st.markdown("### 📊 Model Information")
    st.markdown("""
        <div class="info-box" style="background: rgba(0, 255, 136, 0.1); border: 2px solid rgba(0, 255, 136, 0.3);">
            <p style="font-size: 0.85rem; margin: 0.3rem 0;">
                <b>Coarse Model:</b> VulBERTa (Function-level)<br>
                <b>Fine Model:</b> LineVul (Line-level)<br>
                <b>Training Data:</b> BigVul Dataset<br>
                <b>Task:</b> Binary Classification<br>
                <b>Device:</b> CPU<br>
                <b>Coarse Accuracy:</b> 94.15%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Limitations Section
    with st.expander("⚠️ Important Limitations"):
        st.warning("""
**This model has limitations:**
- Analyzes **per-line**, not cross-line context
- Trained on C/Java (may not work well for Python/JS)
- Pattern-based, not semantic understanding
- Prone to false positives and false negatives
- **Should NOT be used alone for production security decisions**
        """)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">ℹ️ About VulBERT</h4>
            <p style="font-size: 0.9rem;">
                VulBERT is an AI model trained to detect security vulnerabilities in code.
                It analyzes both function-level and line-level patterns.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">🎯 Detection Types</h4>
            <ul style="font-size: 0.9rem; padding-left: 1.5rem;">
                <li>Buffer Overflows (strcpy, gets)</li>
                <li>Use-After-Free</li>
                <li>Integer Overflows</li>
                <li>Format String Issues</li>
                <li>Memory Safety Issues</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">✅ Best Practices</h4>
            <ul style="font-size: 0.9rem; padding-left: 1.5rem;">
                <li>Use as <b>first-pass scanner</b></li>
                <li><b>Review flagged lines</b> manually</li>
                <li>Combine with <b>static analysis</b> tools</li>
                <li>Always perform <b>manual code review</b></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Debug Section
    with st.expander("🔧 Debug & Validation Tools"):
        st.markdown("### Model Debugging")
        
        if st.button("🔍 Debug Line Model", help="Run detailed debugging on line-level model predictions"):
            with st.spinner("Running line model debug..."):
                debug_results = debug_line_model_predictions(
                    st.session_state.line_model, 
                    st.session_state.tokenizer, 
                    st.session_state.device
                )
                
                st.markdown("#### 📊 Debug Results")
                for result in debug_results:
                    risk_color = "🔴" if result['prediction'] == 'VULNERABLE' else "🟢"
                    st.write(f"{risk_color} **{result['prediction']}** ({result['confidence']:.1%})")
                    st.code(result['line'], language="c")
                    st.write(f"SAFE: {result['prob_safe']:.3f} | VULNERABLE: {result['prob_vul']:.3f}")
                    st.markdown("---")
        
        if st.button("✅ Validate Line Model", help="Run comprehensive validation on line-level model"):
            with st.spinner("Running line model validation..."):
                validation_results = validate_line_model(
                    st.session_state.line_model,
                    st.session_state.tokenizer,
                    st.session_state.device
                )
                
                st.markdown("#### 📈 Validation Summary")
                summary = validation_results["summary"]
                st.metric("Overall Accuracy", f"{summary['accuracy']:.1%}")
                st.metric("Correct Predictions", f"{summary['correct_predictions']}/{summary['total_tests']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Cases", summary['high_risk_cases'])
                with col2:
                    st.metric("Medium Risk Cases", summary['medium_risk_cases'])
                with col3:
                    st.metric("Low Risk Cases", summary['low_risk_cases'])
                
                st.markdown("#### 📋 Detailed Results")
                for risk_level in ["high", "medium", "low"]:
                    if validation_results[risk_level]:
                        st.markdown(f"**{risk_level.upper()} Risk Cases:**")
                        for item in validation_results[risk_level]:
                            status = "✅" if item['correct'] else "❌"
                            st.write(f"{status} {item['description']}")
                            st.caption(f"Code: `{item['code']}` | Score: {item['prob_vul']:.3f} | Expected: {item['expected']} | Got: {item['actual']}")
                        st.markdown("---")
        
        if st.button("🔄 Test Model Loading", help="Test standalone model loading functionality"):
            with st.spinner("Testing standalone model loading..."):
                try:
                    test_coarse, test_line, test_tokenizer, test_device = load_models_standalone()
                    if test_coarse and test_line:
                        st.success("✅ Standalone model loading successful!")
                        st.write(f"Coarse model: {type(test_coarse).__name__}")
                        st.write(f"Line model: {type(test_line).__name__}")
                        st.write(f"Device: {test_device}")
                        st.write(f"Tokenizer vocab size: {len(test_tokenizer)}")
                    else:
                        st.error("❌ Standalone model loading failed")
                except Exception as e:
                    st.error(f"❌ Error during standalone loading: {e}")
        
        # Show current model validation status
        if 'validation_result' in st.session_state:
            val_result = st.session_state.validation_result
            st.markdown("#### 📊 Current Model Status")
            status_color = "🟢" if val_result["valid"] else "🟡"
            st.write(f"{status_color} **Coarse Model Validation:** {val_result['accuracy']:.1%} accuracy")
            
            if not val_result["valid"]:
                st.warning("Model validation failed - results may be unreliable")
                with st.expander("View validation details"):
                    for detail in val_result["details"]:
                        status_icon = "✅" if detail["correct"] else "❌"
                        st.write(f"{status_icon} {detail['description']}: {['SAFE', 'VULNERABLE'][detail['predicted']]} (conf: {detail['confidence']:.1%})")

# Load models
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading VulBERT models..."):
        coarse_model, line_model, tokenizer, device = load_models()
        if coarse_model is not None and line_model is not None:
            # Validate coarse model is properly trained
            with st.spinner("🔍 Validating coarse model accuracy..."):
                validation_result = validate_model(coarse_model, tokenizer, device)
            
            if validation_result["valid"]:
                st.session_state.coarse_model = coarse_model
                st.session_state.line_model = line_model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.session_state.model_valid = True
                st.session_state.validation_result = validation_result
                st.success(f"✅ Models loaded and validated! (Coarse test accuracy: {validation_result['accuracy']:.0%})")
            else:
                st.warning(f"⚠️ Coarse model validation WARNING: Only {validation_result['accuracy']:.0%} accuracy on test cases")
                st.info("The coarse model may not be properly trained. Details:")
                for detail in validation_result["details"]:
                    status = "✓" if detail["correct"] else "✗"
                    st.write(f"{status} {detail['description']}: {['SAFE', 'VULNERABLE'][detail['predicted']]}")
                st.session_state.coarse_model = coarse_model
                st.session_state.line_model = line_model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.session_state.model_valid = False
                st.session_state.validation_result = validation_result
        else:
            st.error("❌ Failed to load models. Please check the model paths.")
            st.stop()

# Main content
col1, col2 = st.columns([1.5, 1])

# Define examples dictionary
examples = {
    "Unsafe strcpy": """int unsafe_copy(char *src) {
    char buf[32];
    strcpy(buf, src);  // VULNERABLE: buffer overflow
    return 0;
}""",
    "Safe strncpy": """int safe_copy(const char *src, char *dst, size_t size) {
    if (size == 0) return 0;
    strncpy(dst, src, size - 1);  // SAFE: bounded copy
    dst[size - 1] = '\\0';
    return 0;
}""",
    "Use-after-free": """void process(int *ptr) {
    free(ptr);
    int val = *ptr;  // USE-AFTER-FREE
    printf("%d", val);
}""",
    "Integer Overflow": """int add(int a, int b) {
    int sum = a + b;  // May overflow if a+b > INT_MAX
    return sum;
}"""
}

with col1:
    st.markdown("### 📝 Code Input")
    
    # Initialize example code in session state if not exists
    if 'example_code' not in st.session_state:
        st.session_state.example_code = ""
    
    code_input = st.text_area(
        "Enter your code here:",
        value=st.session_state.example_code,
        placeholder="Paste C, C++, Java, or Python code...",
        height=300,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎮 Controls")
    
    example_select = st.selectbox(
        "📚 Load Example:",
        options=[
            "None",
            "Unsafe strcpy",
            "Safe strncpy",
            "Use-after-free",
            "Integer Overflow"
        ]
    )
    
    if example_select != "None":
        if st.button("📥 Load Example"):
            st.session_state.example_code = examples[example_select]
            st.rerun()
# Analyze button
if st.button("🔍 Analyze Code", use_container_width=True):
    if not code_input.strip():
        st.warning("⚠️ Please enter some code to analyze!")
    else:
        with st.spinner("🔄 Analyzing code..."):
            # Get predictions
            coarse_result = predict_coarse(
                code_input,
                st.session_state.coarse_model,
                st.session_state.tokenizer,
                st.session_state.device
            )

            line_result = predict_lines(
                code_input,
                st.session_state.line_model,
                st.session_state.tokenizer,
                st.session_state.device
            )

            # Display line-level results only
            st.markdown("---")
            st.markdown("### 🔦 Line-Level Vulnerability Analysis")

            st.markdown("#### 📌 Annotated Code")
            render_code_with_highlighting(code_input, line_result['all_lines'], threshold)

            # Filter suspicious lines
            suspicious_lines = [l for l in line_result['all_lines'] if l["prob_vul"] > threshold]

            if suspicious_lines:
                st.markdown("---")
                st.markdown("#### 🔍 Top Suspicious Lines (Above Threshold)")
                for l in suspicious_lines:
                    st.markdown(f"""
                        <div class="metric-card">
                            <b>⚠️ Line {l['line_no']}</b> — <span style="color:#ff3366;">{l['prob_vul']*100:.1f}% risk</span><br>
                            <code>{l['code']}</code>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="safe-card" style="text-align:center;">
                        ✅ No significantly suspicious lines detected.
                    </div>
                """, unsafe_allow_html=True)

    # Debug output section (optional, only visible if show_all_lines enabled)
    if show_all_lines:
        st.markdown("---")
        st.markdown("### 🔧 Debug Output")

        st.markdown("#### 📋 All Line Predictions")
        debug_df = []
        for line_info in line_result['all_lines']:
            debug_df.append({
                'Line #': line_info['line_no'],
                'Code': line_info['code'][:50] + ('...' if len(line_info['code']) > 50 else ''),
                'Vul Prob': f"{line_info['prob_vul']:.3f}",
                'Risk Level': '🔴 High' if line_info['prob_vul'] > threshold else '🟠 Medium' if line_info['prob_vul'] > 0.3 else '🟢 Low'
            })

        if debug_df:
            import pandas as pd
            st.dataframe(pd.DataFrame(debug_df), use_container_width=True)
  
# # Analyze button
# if st.button("🔍 Analyze Code", use_container_width=True):
#     if not code_input.strip():
#         st.warning("⚠️ Please enter some code to analyze!")
#     else:
#         with st.spinner("🔄 Analyzing code..."):
#             # Get predictions
#             coarse_result = predict_coarse(
#                 code_input,
#                 st.session_state.coarse_model,
#                 st.session_state.tokenizer,
#                 st.session_state.device
#             )
            
#             line_result = predict_lines(
#                 code_input,
#                 st.session_state.line_model,
#                 st.session_state.tokenizer,
#                 st.session_state.device
#             )
            
#             # Display results
#             st.markdown("---")
#             st.markdown("### 📊 Analysis Results")
            
#             # Model status warning if not fully validated
#             if not st.session_state.get('model_valid', False):
#                 st.warning("""
#                     ⚠️ **Model Validation Warning:** The model did not achieve full accuracy on validation tests.
#                     Results should be reviewed carefully and combined with other security tools.
#                 """)
            
#             # Function-level result
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 prediction = coarse_result['prediction']
#                 if prediction == 'Vulnerable':
#                     st.markdown(f"""
#                         <div class="metric-card">
#                             <div style="text-align: center;">
#                                 <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚠️</div>
#                                 <div class="prediction-badge vulnerable-badge">VULNERABLE</div>
#                                 <div style="color: rgba(255, 51, 102, 0.8); margin-top: 1rem; font-size: 0.9rem;">
#                                     This function may contain security issues
#                                 </div>
#                             </div>
#                         </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                         <div class="safe-card">
#                             <div style="text-align: center;">
#                                 <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✅</div>
#                                 <div class="prediction-badge safe-badge">SAFE</div>
#                                 <div style="color: rgba(0, 255, 136, 0.8); margin-top: 1rem; font-size: 0.9rem;">
#                                     No obvious security issues detected
#                                 </div>
#                             </div>
#                         </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div style="text-align: center;">
#                             <div style="font-size: 0.9rem; color: rgba(0, 217, 255, 0.7); margin-bottom: 0.5rem;">SAFE CONFIDENCE</div>
#                             <div class="metric-value">{coarse_result['prob_safe']:.1%}</div>
#                             <div class="confidence-bar">
#                                 <div class="confidence-fill" style="width: {coarse_result['prob_safe']*100}%;"></div>
#                             </div>
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
            
#             with col3:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div style="text-align: center;">
#                             <div style="font-size: 0.9rem; color: rgba(255, 51, 102, 0.7); margin-bottom: 0.5rem;">VULNERABLE CONFIDENCE</div>
#                             <div class="metric-value">{coarse_result['prob_vulnerable']:.1%}</div>
#                             <div class="confidence-bar">
#                                 <div class="confidence-fill" style="width: {coarse_result['prob_vulnerable']*100}%;"></div>
#                             </div>
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
            
#             # Line-level analysis
#             st.markdown("---")
#             st.markdown("### 🔦 Line-Level Analysis")
            
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 st.markdown("#### 📌 Annotated Code")
#                 render_code_with_highlighting(code_input, line_result['all_lines'], threshold)
            
#             with col2:
#                 st.markdown("#### 🎯 Risk Breakdown")
                
#                 all_lines = line_result['all_lines']
#                 sorted_lines = line_result['sorted_lines']
                
#                 total_lines = len(all_lines)
#                 high_risk = sum(1 for s in all_lines if s['prob_vul'] > threshold)
#                 medium_risk = sum(1 for s in all_lines if 0.3 < s['prob_vul'] <= threshold)
#                 low_risk = total_lines - high_risk - medium_risk
                
#                 st.metric("🔴 High Risk", high_risk)
#                 st.metric("🟠 Medium Risk", medium_risk)
#                 st.metric("🟢 Low Risk", low_risk)
                
#                 st.markdown("---")
#                 st.markdown("#### 📈 Top Suspicious Lines")
                
#                 for idx, line_info in enumerate(sorted_lines[:5], 1):
#                     risk_level = "🔴 High" if line_info['prob_vul'] > threshold else "🟠 Medium" if line_info['prob_vul'] > 0.3 else "🟢 Low"
#                     st.caption(f"{idx}. Line {line_info['line_no']} - {risk_level}")
#                     st.code(line_info['code'][:60] + ("..." if len(line_info['code']) > 60 else ""), language="c")
#                     st.caption(f"Risk Score: {line_info['prob_vul']:.1%}")
    
#     # Debug output section (only show if debug mode is enabled)
#     if show_all_lines:
#         st.markdown("---")
#         st.markdown("### 🔧 Debug Output")
        
#         st.markdown("#### 📋 All Line Predictions")
#         debug_df = []
#         for line_info in line_result['all_lines']:
#             debug_df.append({
#                 'Line #': line_info['line_no'],
#                 'Code': line_info['code'][:50] + ('...' if len(line_info['code']) > 50 else ''),
#                 'Vul Prob': f"{line_info['prob_vul']:.3f}",
#                 'Risk Level': '🔴 High' if line_info['prob_vul'] > threshold else '🟠 Medium' if line_info['prob_vul'] > 0.3 else '🟢 Low'
#             })
        
#         if debug_df:
#             import pandas as pd
#             st.dataframe(pd.DataFrame(debug_df), use_container_width=True)
        
        st.markdown("#### 📊 Model Raw Outputs")
        st.write("**Coarse Model Result:**")
        st.json(coarse_result)
        
        st.write("**Line Model Summary:**")
        st.write(f"- Total lines analyzed: {len(line_result['all_lines'])}")
        st.write(f"- Lines above threshold ({threshold:.2f}): {sum(1 for s in line_result['all_lines'] if s['prob_vul'] > threshold)}")
        st.write(f"- Average vulnerability score: {sum(s['prob_vul'] for s in line_result['all_lines']) / len(line_result['all_lines']):.3f}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: rgba(0, 217, 255, 0.5); font-size: 0.85rem; padding: 2rem 0;">
        <p>🛡️ VulBERT - Powered by CodeBERT & Transformers</p>
        <p>For research and educational purposes only. Always conduct thorough security reviews.</p>
    </div>
""", unsafe_allow_html=True)
