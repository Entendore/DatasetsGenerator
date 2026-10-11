#!/usr/bin/env python3
"""
Local Dataset Generator (ML Pro Edition) - Improved
=====================================================
Improvements over original:
- BUG FIX: Fixed missing model_combo_oai / url_edit_oai references causing
  OpenAI-compatible backend to fail at runtime.
- Persistence: Window geometry/state and splitter sizes are saved between runs.
- Drag & Drop: Drop a .gguf file onto the model path field to load it.
- File Watcher: Wildcards list auto-refreshes when files change on disk.
- Dataset Statistics: Viewer shows per-field value counts, empty-rate, etc.
- Filter & Search: Dataset viewer has a filter box to search items by content.
- Export Formats: Added ShareGPT and Alpaca JSON exports for ML fine-tuning.
- Robust JSON Extraction: Handles nested objects/arrays, partial outputs, and concatenated objects.
- Timeout Settings: Configurable HTTP timeout (default 600s).
- Continue-on-Error: Toggle to skip broken items instead of failing batches.
- API Key Security: Stored via QSettings with hidden echo mode.
- Tooltips: Helpful tooltips on every control.
- Empty model warning: Falls back to default if model not selected.
- Better Error Messages: Network/parse errors are user-friendly.
- additionalProperties Toggle: Allow flexible schema for reasoning models.
- Fast Connection Test: GGUF validity is checked via magic bytes instead of full load.
- UI Enhancements: Save log to file, copy viewer item to clipboard.
"""

import sys
import os
import json
import re
import random
import csv
import hashlib
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QSettings, QTimer, QFileSystemWatcher, QMimeData
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QComboBox, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QSplitter, QProgressBar, QPlainTextEdit,
    QHeaderView, QTabWidget, QStackedWidget, QListWidget, QListWidgetItem,
    QDialog, QTextEdit, QInputDialog, QMenu
)
from PySide6.QtGui import QFont, QGuiApplication, QAction, QDragEnterEvent, QDropEvent

try:
    import requests
except ImportError:
    requests = None

# --------------------------------------------------------------------------
#  Dark Theme Stylesheet
# --------------------------------------------------------------------------
DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 5px 12px;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border: 1px solid #5a5a5a;
}
QPushButton:pressed { background-color: #333333; }
QPushButton:disabled { color: #666666; background-color: #2f2f2f; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #333333;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 3px;
    color: #e0e0e0;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4a90d9;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #333333;
    border: 1px solid #4a4a4a;
    selection-background-color: #4a90d9;
}
QTableWidget {
    background-color: #333333;
    gridline-color: #4a4a4a;
    border: 1px solid #4a4a4a;
    selection-background-color: #4a90d9;
}
QHeaderView::section {
    background-color: #3c3c3c;
    padding: 4px;
    border: 1px solid #4a4a4a;
    color: #e0e0e0;
}
QTabWidget::pane { border: 1px solid #4a4a4a; background-color: #2b2b2b; }
QTabBar::tab {
    background-color: #333333;
    padding: 8px 16px;
    border: 1px solid #4a4a4a;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #4a4a4a; }
QListWidget {
    background-color: #333333;
    border: 1px solid #4a4a4a;
    selection-background-color: #4a90d9;
}
QProgressBar {
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    text-align: center;
    background-color: #333333;
    color: #e0e0e0;
}
QProgressBar::chunk { background-color: #4a90d9; width: 10px; }
QLabel { background: transparent; }
QSplitter::handle { background-color: #3c3c3c; }
QSplitter::handle:hover { background-color: #4a90d9; }
QToolTip {
    background-color: #333333;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    padding: 4px;
}
"""

# --------------------------------------------------------------------------
#  Regex & Hints
# --------------------------------------------------------------------------
WILDCARD_REGEX = re.compile(r"__([a-zA-Z0-9_\-]+)__")

DIVERSITY_HINTS = [
    "Focus on a highly realistic, mundane scenario.",
    "Introduce a slight edge-case or unusual perspective.",
    "Make the tone slightly more formal and academic.",
    "Make the tone very casual and colloquial.",
    "Focus on a specific, niche sub-topic.",
    "Introduce a minor complication or conflict.",
    "Use a highly descriptive and verbose style.",
    "Keep it very brief and to the point.",
    "Imagine the context is from 50 years in the future.",
    "Imagine the context is from a rural, traditional setting.",
    "Add a subtle sense of humor or irony.",
    "Ensure all entities and names used are highly unique and uncommon."
]

DEFAULT_OLLAMA_MODELS = ["llama3.1:8b", "llama3.1:70b", "mistral-nemo", "qwen2.5:14b"]
DEFAULT_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "local-model"]

# --------------------------------------------------------------------------
#  Model Clients
# --------------------------------------------------------------------------

class ModelClient:
    def __init__(self, backend, base_url, model, temperature, max_tokens,
                 api_key="", timeout=600):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.timeout = timeout

    def generate_json(self, system, user_prompt, json_schema=None):
        if self.backend == "ollama":
            return self._ollama(system, user_prompt, json_schema)
        return self._openai_compat(system, user_prompt, json_schema)

    def _ollama(self, system, user_prompt, json_schema):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "format": json_schema if json_schema else "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s. Try increasing the timeout or reducing batch size.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}. Response: {r.text[:300]}")
        return r.json()["message"]["content"]

    def _openai_compat(self, system, user_prompt, json_schema):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},  # Safe default for broad compatibility
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload, headers=headers, timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Server request timed out after {self.timeout}s.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to {self.base_url}. Is the server running?")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP error {e}. Response: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


class LocalLlamaCppClient:
    def __init__(self, model_path, n_ctx, n_gpu_layers, n_threads,
                 temperature, max_tokens):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "The 'llama-cpp-python' library is required for local GGUF models.\n"
                "Install it via:  pip install llama-cpp-python\n"
                "Or for GPU:  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_json(self, system, user_prompt, json_schema=None):
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object", "schema": json_schema}
                           if json_schema else {"type": "json_object"},
            stream=False,
        )
        return response["choices"][0]["message"]["content"]


def list_models(backend, base_url, api_key=""):
    base_url = base_url.rstrip("/")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if backend == "ollama":
            r = requests.get(f"{base_url}/api/tags", timeout=10, headers=headers)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        else:
            r = requests.get(f"{base_url}/v1/models", timeout=10, headers=headers)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []

# --------------------------------------------------------------------------
#  Drag-Drop GGUF LineEdit
# --------------------------------------------------------------------------

class GGUFLineEdit(QLineEdit):
    """QLineEdit that accepts .gguf files dragged onto it."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drop a .gguf file here or click Browse…")

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(".gguf") for u in urls):
                e.acceptProposedAction()
                return
        e.ignore()

    def dropEvent(self, e: QDropEvent):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".gguf"):
                self.setText(p)
                e.acceptProposedAction()
                return

# --------------------------------------------------------------------------
#  Worker Thread
# --------------------------------------------------------------------------

class GenerationWorker(QObject):
    progress = Signal(int, int)
    item_generated = Signal(int, dict)
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, fields, instruction, examples, custom_system_prompt,
                 n_items, batch_size, client_config, output_dir, dataset_name,
                 delay, append, wildcard_dir, deduplicate, split_ratios,
                 allow_extra_properties=False, continue_on_error=True):
        super().__init__()
        self.fields = fields
        self.instruction = instruction
        self.examples = examples
        self.custom_system_prompt = custom_system_prompt
        self.n_items = n_items
        self.batch_size = batch_size
        self.client_config = client_config
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.delay = delay
        self.append = append
        self.wildcard_dir = Path(wildcard_dir)
        self.wildcards_cache = {}
        self.deduplicate = deduplicate
        self.split_ratios = split_ratios
        self.allow_extra_properties = allow_extra_properties
        self.continue_on_error = continue_on_error
        self._stop = False
        self.client = None
        self.seen_hashes = set()
        self.failures = 0

    def stop(self):
        self._stop = True

    def load_wildcards(self):
        self.wildcards_cache = {}
        if not self.wildcard_dir.exists(): return

        for f in self.wildcard_dir.glob("*.txt"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    lines, weights = [], []
                    for line in file.readlines():
                        line = line.strip()
                        if not line or line.startswith("#"): continue
                        if '::' in line:
                            parts = line.split('::', 1)
                            try:
                                weights.append(int(parts[0]))
                                lines.append(parts[1].strip())
                            except ValueError:
                                weights.append(1)
                                lines.append(line)
                        else:
                            weights.append(1)
                            lines.append(line)
                if lines:
                    self.wildcards_cache[f.stem] = (lines, weights)
            except Exception: pass

        if self.wildcards_cache:
            self.log.emit(f"Loaded {len(self.wildcards_cache)} wildcard file(s).")

    def parse_wildcards(self, text):
        if not text: return text
        max_depth = 10
        for _ in range(max_depth):
            if '__' not in text: break
            def replacer(match):
                wc_name = match.group(1)
                if wc_name in self.wildcards_cache:
                    lines, weights = self.wildcards_cache[wc_name]
                    return random.choices(lines, weights=weights, k=1)[0]
                return match.group(0)
            new_text = WILDCARD_REGEX.sub(replacer, text)
            if new_text == text: break
            text = new_text
        return text

    def build_json_schema(self, batch=False):
        type_map = {"string": "string", "integer": "integer", "float": "number",
                    "boolean": "boolean", "list": "array", "object": "object"}
        properties, required = {}, []
        for f in self.fields:
            t = type_map.get(f["type"], "string")
            desc = self.parse_wildcards(f["description"])
            prop = {"type": t, "description": desc}
            if t == "array": prop["items"] = {"type": "string"}
            properties[f["name"]] = prop
            required.append(f["name"])

        item_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": self.allow_extra_properties,
        }
        if batch:
            return {"type": "array", "items": item_schema,
                    "minItems": self.batch_size, "maxItems": self.batch_size}
        return item_schema

    def build_system_prompt(self, json_schema):
        if self.custom_system_prompt.strip():
            return self.parse_wildcards(self.custom_system_prompt)

        schema_str = json.dumps(json_schema, indent=2)
        lines = [
            "You are a dataset generator. Produce realistic, coherent data records.",
            "Output ONLY valid JSON that strictly matches the JSON Schema below.",
            "Do not include markdown fences, explanations, or any other text.",
            "", "JSON Schema:", schema_str, "",
        ]
        parsed_instruction = self.parse_wildcards(self.instruction)
        if parsed_instruction.strip():
            lines += ["Additional instructions:", parsed_instruction.strip(), ""]

        parsed_examples = self.parse_wildcards(self.examples)
        if parsed_examples.strip():
            lines += ["Here are example(s) of the desired output format and style:",
                      parsed_examples.strip(), ""]

        lines.append("You must strictly obey the schema.")
        return "\n".join(lines)

    def build_user_prompt(self, idx, count=1):
        hint = random.choice(DIVERSITY_HINTS)
        parsed_hint = self.parse_wildcards(hint)
        if count > 1:
            return (f"Generate {count} distinct items for the dataset "
                    f"(starting at index {idx + 1}). To ensure diversity, {parsed_hint} "
                    f"Return ONLY a JSON array containing the {count} objects.")
        return (f"Generate item #{idx + 1} of the dataset. To ensure diversity, {parsed_hint} "
                f"Return only the JSON object.")

    @staticmethod
    def extract_json(text):
        """Improved JSON extraction. Handles fenced code, raw JSON, and partial outputs."""
        if not text:
            return None, "empty response"

        # 1) Direct parse
        try:
            return json.loads(text), None
        except Exception:
            pass

        # 2) Strip markdown fences ```json ... ``` or ``` ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)), None
            except Exception:
                pass

        # 3) Find the first balanced JSON object/array via bracket matching
        def find_balanced(s, open_ch, close_ch):
            depth, start = 0, -1
            in_str = False
            esc = False
            for i, c in enumerate(s):
                if in_str:
                    if esc: esc = False
                    elif c == '\\': esc = True
                    elif c == '"': in_str = False
                else:
                    if c == '"': in_str = True
                    elif c == open_ch:
                        if depth == 0: start = i
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0 and start >= 0:
                            return s[start:i+1]
            return None

        # Try to find an array first
        arr = find_balanced(text, '[', ']')
        if arr:
            try:
                return json.loads(arr), None
            except Exception:
                pass
        
        # Try to find an object
        obj = find_balanced(text, '{', '}')
        if obj:
            try:
                return json.loads(obj), None
            except Exception:
                pass

        # 4) Handle concatenated objects (e.g. `{...} {...}`)
        objs = re.findall(r'\{.*?\}', text, re.DOTALL)
        if len(objs) > 1:
            try:
                wrapped = "[" + ",".join(objs) + "]"
                return json.loads(wrapped), None
            except Exception as e:
                return None, f"failed to parse concatenated objects: {e}"

        return None, "no JSON object found in response"

    def sanitize_item(self, item):
        if not isinstance(item, dict): return None
        clean = {}
        for f in self.fields:
            name, typ = f["name"], f["type"]
            if name not in item:
                clean[name] = [] if typ == "list" else None
                continue
            val = item[name]
            try:
                if typ == "integer":
                    clean[name] = int(val) if not isinstance(val, bool) else int(val)
                elif typ == "float":
                    clean[name] = float(val)
                elif typ == "boolean":
                    clean[name] = bool(val) if isinstance(val, bool) else str(val).lower() in ('true', '1', 'yes')
                elif typ == "list":
                    if isinstance(val, list): clean[name] = val
                    elif isinstance(val, str): 
                        try: clean[name] = json.loads(val)
                        except: clean[name] = [v.strip() for v in val.split(',')]
                    else: clean[name] = [val] if val is not None else []
                elif typ == "object":
                    if isinstance(val, dict): clean[name] = val
                    elif isinstance(val, str):
                        try: clean[name] = json.loads(val)
                        except: clean[name] = {"value": val}
                    else: clean[name] = {"value": val}
                elif typ == "string":
                    if isinstance(val, str): clean[name] = val
                    else: clean[name] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
                else:
                    clean[name] = val
            except Exception:
                clean[name] = str(val) if val is not None else None
        return clean

    def init_client(self):
        cfg = self.client_config
        if cfg["backend"] == "local-llama-cpp":
            self.log.emit("Loading local GGUF model into memory...")
            self.client = LocalLlamaCppClient(
                model_path=cfg["model_path"], n_ctx=cfg["n_ctx"],
                n_gpu_layers=cfg["n_gpu_layers"], n_threads=cfg["n_threads"],
                temperature=cfg["temperature"], max_tokens=cfg["max_tokens"]
            )
            self.log.emit("Model loaded successfully.")
        else:
            self.client = ModelClient(
                backend=cfg["backend"], base_url=cfg["base_url"],
                model=cfg["model"], temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                api_key=cfg.get("api_key", ""),
                timeout=cfg.get("timeout", 600),
            )

    def split_dataset(self, ds_dir):
        train_pct, test_pct, val_pct = self.split_ratios
        if train_pct == 100: return

        jsonl_path = ds_dir / "items.jsonl"
        if not jsonl_path.exists(): return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]

        random.shuffle(items)
        n = len(items)
        n_train = int(n * (train_pct / 100.0))
        n_test = int(n * (test_pct / 100.0))

        train_items = items[:n_train]
        test_items = items[n_train:n_train+n_test]
        val_items = items[n_train+n_test:]

        def write_jsonl(filepath, data):
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(item + "\n")

        write_jsonl(ds_dir / "train.jsonl", train_items)
        write_jsonl(ds_dir / "test.jsonl", test_items)
        write_jsonl(ds_dir / "val.jsonl", val_items)

        self.log.emit(f"Split: Train({len(train_items)}) / Test({len(test_items)}) / Val({len(val_items)})")

    def run(self):
        if requests is None and self.client_config["backend"] != "local-llama-cpp":
            self.finished.emit(False, "The 'requests' library is required for server backends. Install via 'pip install requests'.")
            return

        try:
            self.load_wildcards()
            self.init_client()

            ds_dir = Path(self.output_dir) / self.dataset_name
            items_dir = ds_dir / "items"
            items_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = ds_dir / "items.jsonl"

            existing = sorted(items_dir.glob("*.json"))
            start_idx = len(existing) if self.append else 0
            if not self.append and existing:
                for p in existing: p.unlink()
                if jsonl_path.exists(): jsonl_path.unlink()

            if self.deduplicate and jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        item_str = line.strip()
                        if item_str:
                            h = hashlib.md5(item_str.encode('utf-8')).hexdigest()
                            self.seen_hashes.add(h)

            is_batch = self.batch_size > 1
            json_schema = self.build_json_schema(batch=is_batch)
            system_prompt = self.build_system_prompt(json_schema)

            self.log.emit("=== System prompt (Sample, wildcards randomized per call) ===")
            self.log.emit(system_prompt)
            self.log.emit("=========================================================\n")

            mode = "a" if self.append else "w"
            successes = 0
            generated_count = 0
            consec_failures = 0
            MAX_CONSEC_FAILURES = 5

            with open(jsonl_path, mode, encoding="utf-8") as jsonl_file:
                while generated_count < self.n_items:
                    if self._stop:
                        self.log.emit("Stopped by user.")
                        break

                    current_batch_size = min(self.batch_size, self.n_items - generated_count)
                    idx = start_idx + generated_count

                    # Wildcards re-parsed per call for diversity
                    system_prompt = self.build_system_prompt(json_schema)
                    user_prompt = self.build_user_prompt(idx, current_batch_size)

                    max_retries = 3
                    parsed_data = None
                    last_err = ""
                    for attempt in range(max_retries):
                        try:
                            self.log.emit(f"[{idx + 1}] Requesting {current_batch_size} item(s) (Attempt {attempt + 1}/{max_retries})...")
                            raw = self.client.generate_json(system_prompt, user_prompt, json_schema)
                            parsed_data, err = self.extract_json(raw)
                            if parsed_data is not None: break
                            last_err = err
                            self.log.emit(f"[{idx + 1}] Parse error: {err}. Retrying...")
                        except RuntimeError as e:
                            last_err = str(e)
                            self.log.emit(f"[{idx + 1}] Network/Server Error: {e}. Retrying...")
                            QThread.msleep(2000)
                        except Exception as e:
                            last_err = str(e)
                            self.log.emit(f"[{idx + 1}] Unexpected Error: {e}. Retrying...")
                            QThread.msleep(1000)

                    if parsed_data is None:
                        self.log.emit(f"[{idx + 1}] Failed after {max_retries} attempts ({last_err}). Skipping batch.")
                        self.failures += current_batch_size
                        generated_count += current_batch_size
                        consec_failures += 1
                        if consec_failures >= MAX_CONSEC_FAILURES and not self.continue_on_error:
                            self.log.emit(f"Aborting: {consec_failures} consecutive batch failures.")
                            break
                        continue
                    consec_failures = 0

                    items = parsed_data if isinstance(parsed_data, list) else [parsed_data]

                    for i, raw_item in enumerate(items):
                        if generated_count >= self.n_items: break
                        item_idx = start_idx + generated_count
                        item = self.sanitize_item(raw_item)
                        if not item:
                            self.log.emit(f"[{item_idx + 1}] Invalid item, skipping...")
                            continue

                        item_str = json.dumps(item, ensure_ascii=False)
                        item_hash = hashlib.md5(item_str.encode('utf-8')).hexdigest()

                        if self.deduplicate and item_hash in self.seen_hashes:
                            self.log.emit(f"[{item_idx + 1}] Duplicate detected. Skipping.")
                            continue

                        self.seen_hashes.add(item_hash)

                        fname = items_dir / f"{item_idx + 1:05d}.json"
                        with open(fname, "w", encoding="utf-8") as fp:
                            json.dump(item, fp, indent=2, ensure_ascii=False)

                        jsonl_file.write(item_str + "\n")
                        jsonl_file.flush()

                        successes += 1
                        generated_count += 1
                        self.item_generated.emit(item_idx + 1, item)
                        self.progress.emit(generated_count, self.n_items)
                        self.log.emit(f"[{item_idx + 1}] Saved {fname.name}")

                    if self.delay > 0 and generated_count < self.n_items:
                        QThread.msleep(int(self.delay * 1000))

            meta = {
                "dataset": self.dataset_name,
                "backend": self.client_config["backend"],
                "model": self.client_config.get("model", self.client_config.get("model_path", "unknown")),
                "temperature": self.client_config["temperature"],
                "max_tokens": self.client_config["max_tokens"],
                "items_requested": self.n_items,
                "items_generated": successes,
                "items_failed": self.failures,
                "appended": self.append,
                "deduplicated": self.deduplicate,
                "allow_extra_properties": self.allow_extra_properties,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(ds_dir / "meta.json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, indent=2)
            with open(ds_dir / "format.json", "w", encoding="utf-8") as fp:
                json.dump({"fields": self.fields, "instruction": self.instruction,
                           "examples": self.examples,
                           "custom_system_prompt": self.custom_system_prompt}, fp, indent=2)

            self.split_dataset(ds_dir)

            status = "Completed" if successes == self.n_items else "Completed with some failures"
            self.finished.emit(True,
                f"{status}. Generated {successes}/{self.n_items} items"
                f" ({self.failures} failed) in:\n{ds_dir}")
        except Exception as e:
            self.finished.emit(False, f"Fatal error: {e}")


class PromptPreviewDialog(QDialog):
    def __init__(self, system_prompt, json_schema, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Preview (Wildcards Parsed Once)")
        self.resize(640, 560)
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("JSON Schema:"))
        schema_view = QPlainTextEdit()
        schema_view.setPlainText(json.dumps(json_schema, indent=2))
        schema_view.setReadOnly(True)
        layout.addWidget(schema_view)
        
        layout.addWidget(QLabel("System Prompt:"))
        prompt_view = QPlainTextEdit()
        prompt_view.setPlainText(system_prompt)
        prompt_view.setReadOnly(True)
        layout.addWidget(prompt_view)
        
        btn_layout = QHBoxLayout()
        btn_copy = QPushButton("Copy Prompt")
        btn_copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(system_prompt))
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_copy)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)


FIELD_TYPES = ["string", "integer", "float", "boolean", "list", "object"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Dataset Generator (ML Pro Edition)")
        self.resize(1380, 940)
        self.worker = None
        self.thread = None
        self.settings = QSettings("LocalAI", "DatasetGeneratorMLPro")
        self._build_ui()
        self._load_settings()
        self._restore_window_state()
        self._load_default_fields()
        self._update_viewer_datasets()
        self.refresh_wildcard_list()
        self._setup_wildcard_watcher()
        self._update_dataset_stats()

    # ----------------------------------------------------------------------
    #  UI build
    # ----------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self._main_splitter = QSplitter(Qt.Vertical)
        root.addWidget(self._main_splitter, 1)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(4, 4, 4, 4)
        self._top_splitter = QSplitter(Qt.Horizontal)
        top_layout.addWidget(self._top_splitter, 1)

        # ---- LEFT: Format editor ----
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)

        fmt_box = QGroupBox("Dataset Format")
        fbl = QVBoxLayout(fmt_box)

        self.fields_table = QTableWidget(0, 4)
        self.fields_table.setHorizontalHeaderLabels(["Field Name", "Type", "Description", ""])
        self.fields_table.setToolTip("Define the schema fields. Wildcards __name__ are supported in the Description column.")
        hdr = self.fields_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        fbl.addWidget(self.fields_table)

        fb = QHBoxLayout()
        b_add = QPushButton("+ Add Field"); b_add.clicked.connect(lambda: self.add_field_row())
        b_rm = QPushButton("− Remove Selected"); b_rm.clicked.connect(self.remove_selected_field)
        b_clear = QPushButton("Clear All"); b_clear.clicked.connect(self.clear_fields)
        fb.addWidget(b_add); fb.addWidget(b_rm); fb.addWidget(b_clear); fb.addStretch(1)
        fbl.addLayout(fb)

        adv_box = QGroupBox("Advanced Prompting")
        adv_layout = QVBoxLayout(adv_box)

        adv_layout.addWidget(QLabel("Instruction (supports __wildcards__):"))
        self.instruction_edit = QPlainTextEdit()
        self.instruction_edit.setToolTip("Free-form instructions added to the system prompt. Wildcards supported.")
        self.instruction_edit.setMaximumHeight(50)
        adv_layout.addWidget(self.instruction_edit)

        adv_layout.addWidget(QLabel("Few-shot Examples (supports __wildcards__):"))
        self.examples_edit = QPlainTextEdit()
        self.examples_edit.setPlaceholderText('[\n  {"title": "Example", "content": "This is an example."}\n]')
        self.examples_edit.setMaximumHeight(70)
        adv_layout.addWidget(self.examples_edit)

        adv_layout.addWidget(QLabel("Custom System Prompt (Overrides auto-schema prompt):"))
        self.custom_sys_edit = QPlainTextEdit()
        self.custom_sys_edit.setPlaceholderText("Leave empty to use auto-generated prompt. Use __wildcards__ here too.")
        self.custom_sys_edit.setMaximumHeight(70)
        adv_layout.addWidget(self.custom_sys_edit)

        fbl.addWidget(adv_box)

        tb = QHBoxLayout()
        b_save = QPushButton("Save Format…"); b_save.clicked.connect(self.save_format)
        b_load = QPushButton("Load Format…"); b_load.clicked.connect(self.load_format)
        b_preview = QPushButton("Preview Prompt"); b_preview.clicked.connect(self.preview_prompt)
        tb.addWidget(b_save); tb.addWidget(b_load); tb.addWidget(b_preview)
        tb.addStretch(1)
        fbl.addLayout(tb)

        ll.addWidget(fmt_box, 1)
        self._top_splitter.addWidget(left)

        # ---- RIGHT: controls ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()

        # -- Model tab --
        m_tab = QWidget()
        m_form = QFormLayout(m_tab)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["local-llama-cpp", "ollama", "openai-compatible"])
        self.backend_combo.setToolTip("Choose how to call the LLM. local-llama-cpp runs GGUF in-process; the other two connect to a server.")
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.8)
        self.temp_spin.setToolTip("Sampling temperature. Higher = more diverse/random.")

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 32768)
        self.max_tokens_spin.setValue(2048)
        self.max_tokens_spin.setToolTip("Maximum tokens generated per API call (not per item).")

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 7200)
        self.timeout_spin.setSingleStep(30)
        self.timeout_spin.setValue(600)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setToolTip("HTTP timeout per request in seconds.")

        m_form.addRow("Backend:", self.backend_combo)
        m_form.addRow("Temperature:", self.temp_spin)
        m_form.addRow("Max Tokens (per call):", self.max_tokens_spin)
        m_form.addRow("HTTP Timeout:", self.timeout_spin)

        self.backend_stack = QStackedWidget()

        # Page 0: local-llama-cpp
        local_page = QWidget()
        lp_form = QFormLayout(local_page)
        self.model_path_edit = GGUFLineEdit()
        b_browse_gguf = QPushButton("Browse…")
        b_browse_gguf.clicked.connect(self.browse_gguf)
        lp_form.addRow("Model (.gguf):", MainWindow._row(self.model_path_edit, b_browse_gguf))
        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 131072)
        self.n_ctx_spin.setValue(4096)
        lp_form.addRow("Context Size:", self.n_ctx_spin)
        self.n_gpu_spin = QSpinBox()
        self.n_gpu_spin.setRange(-1, 1000)
        self.n_gpu_spin.setValue(0)
        lp_form.addRow("GPU Layers:", self.n_gpu_spin)
        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 128)
        self.n_threads_spin.setValue(8)
        lp_form.addRow("CPU Threads:", self.n_threads_spin)
        self.backend_stack.addWidget(local_page)

        # Page 1: Ollama
        ollama_page = QWidget()
        op_form = QFormLayout(ollama_page)
        self.url_edit = QLineEdit("http://localhost:11434")
        op_form.addRow("Base URL:", self.url_edit)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_OLLAMA_MODELS)
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self.refresh_models)
        op_form.addRow("Model:", MainWindow._row(self.model_combo, b_refresh))
        self.backend_stack.addWidget(ollama_page)

        # Page 2: OpenAI-compatible
        openai_page = QWidget()
        oa_form = QFormLayout(openai_page)
        self.url_edit_oai = QLineEdit("http://localhost:1234")
        oa_form.addRow("Base URL:", self.url_edit_oai)
        self.api_key_edit_oai = QLineEdit()
        self.api_key_edit_oai.setEchoMode(QLineEdit.Password)
        self.api_key_edit_oai.setPlaceholderText("sk-... (Leave empty for local)")
        oa_form.addRow("API Key:", self.api_key_edit_oai)
        self.model_combo_oai = QComboBox()
        self.model_combo_oai.setEditable(True)
        self.model_combo_oai.addItems(DEFAULT_OPENAI_MODELS)
        b_refresh_oai = QPushButton("Refresh")
        b_refresh_oai.clicked.connect(self.refresh_models)
        oa_form.addRow("Model:", MainWindow._row(self.model_combo_oai, b_refresh_oai))
        self.backend_stack.addWidget(openai_page)

        m_form.addRow("Settings:", self.backend_stack)

        b_test = QPushButton("Test Connection / Validate Model")
        b_test.clicked.connect(self.test_connection)
        self.test_label = QLabel("Not tested yet.")
        self.test_label.setWordWrap(True)
        m_form.addRow("", b_test)
        m_form.addRow("", self.test_label)
        tabs.addTab(m_tab, "Model")

        # -- Generation tab --
        g_tab = QWidget()
        g_form = QFormLayout(g_tab)
        self.dataset_name_edit = QLineEdit("dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.base_dir_edit = QLineEdit(str(Path.cwd() / "datasets"))
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self.browse_output)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1_000_000)
        self.count_spin.setValue(50)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 50)
        self.batch_spin.setValue(5)
        self.batch_spin.setToolTip("How many items to request per LLM call. Higher = faster but more likely to hit token limits.")

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 60.0)
        self.delay_spin.setSingleStep(0.25)
        self.delay_spin.setValue(0.0)

        self.append_check = QCheckBox("Append if dataset folder exists")
        self.append_check.setChecked(True)

        self.dedup_check = QCheckBox("Deduplicate items (Exact match)")
        self.dedup_check.setChecked(True)

        self.continue_on_error_check = QCheckBox("Continue on repeated batch failures")
        self.continue_on_error_check.setChecked(True)
        self.continue_on_error_check.setToolTip("If unchecked, generation aborts after 5 consecutive failed batches.")

        self.allow_extra_props_check = QCheckBox("Allow additional properties in schema")
        self.allow_extra_props_check.setChecked(False)
        self.allow_extra_props_check.setToolTip("Set additionalProperties=true. Useful for reasoning models that add metadata.")

        g_form.addRow("Dataset name:", self.dataset_name_edit)
        g_form.addRow("Output base dir:", MainWindow._row(self.base_dir_edit, b_browse))
        g_form.addRow("Total items:", self.count_spin)
        g_form.addRow("Batch size (items/call):", self.batch_spin)
        g_form.addRow("Delay between calls (s):", self.delay_spin)
        g_form.addRow("", self.append_check)
        g_form.addRow("", self.dedup_check)
        g_form.addRow("", self.continue_on_error_check)
        g_form.addRow("", self.allow_extra_props_check)

        # Splitting Group
        split_box = QGroupBox("Dataset Splitting (ML Workflows)")
        split_layout = QHBoxLayout(split_box)
        self.train_spin = QSpinBox(); self.train_spin.setRange(0, 100); self.train_spin.setValue(80)
        self.test_spin = QSpinBox(); self.test_spin.setRange(0, 100); self.test_spin.setValue(10)
        self.val_spin = QSpinBox(); self.val_spin.setRange(0, 100); self.val_spin.setValue(10)
        self.split_btn_check = QCheckBox("Enable Splitting")
        self.split_btn_check.setChecked(False)

        split_layout.addWidget(self.split_btn_check)
        split_layout.addWidget(QLabel("Train %:"))
        split_layout.addWidget(self.train_spin)
        split_layout.addWidget(QLabel("Test %:"))
        split_layout.addWidget(self.test_spin)
        split_layout.addWidget(QLabel("Val %:"))
        split_layout.addWidget(self.val_spin)
        g_form.addRow(split_box)

        tabs.addTab(g_tab, "Generation")

        # -- Wildcards tab --
        wc_tab = QWidget()
        wc_layout = QHBoxLayout(wc_tab)
        wc_splitter = QSplitter(Qt.Horizontal)
        wc_left = QWidget()
        wc_left_layout = QVBoxLayout(wc_left)
        wc_left_layout.addWidget(QLabel("Wildcard Files:"))
        self.wc_list = QListWidget()
        self.wc_list.currentItemChanged.connect(self.load_wildcard_to_editor)
        wc_left_layout.addWidget(self.wc_list)

        wc_btns = QHBoxLayout()
        b_wc_new = QPushButton("New"); b_wc_new.clicked.connect(self.new_wildcard)
        b_wc_del = QPushButton("Delete"); b_wc_del.clicked.connect(self.delete_wildcard)
        b_wc_ref = QPushButton("Refresh"); b_wc_ref.clicked.connect(self.refresh_wildcard_list)
        wc_btns.addWidget(b_wc_new); wc_btns.addWidget(b_wc_del); wc_btns.addWidget(b_wc_ref)
        wc_left_layout.addLayout(wc_btns)
        wc_splitter.addWidget(wc_left)

        wc_right = QWidget()
        wc_right_layout = QVBoxLayout(wc_right)
        wc_right_layout.addWidget(QLabel("Edit Wildcard (Format: weight::value, e.g. 5::happy). Lines starting with # are comments.):"))
        self.wc_editor = QPlainTextEdit()
        wc_right_layout.addWidget(self.wc_editor)
        b_wc_save = QPushButton("Save Wildcard"); b_wc_save.clicked.connect(self.save_wildcard)
        wc_right_layout.addWidget(b_wc_save)
        wc_splitter.addWidget(wc_right)
        wc_splitter.setSizes([220, 420])
        wc_layout.addWidget(wc_splitter)
        tabs.addTab(wc_tab, "Wildcards")

        rl.addWidget(tabs, 0)

        gen_box = QGroupBox("Generate")
        gbl = QVBoxLayout(gen_box)
        self.gen_button = QPushButton("▶  Generate Dataset")
        self.gen_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.gen_button.clicked.connect(self.start_generation)
        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        gb = QHBoxLayout()
        gb.addWidget(self.gen_button); gb.addWidget(self.stop_button)
        gbl.addLayout(gb)
        self.progress_bar = QProgressBar()
        gbl.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready.")
        gbl.addWidget(self.status_label)
        rl.addWidget(gen_box)
        rl.addStretch(1)
        self._top_splitter.addWidget(right)
        self._top_splitter.setSizes([540, 460])

        self._main_splitter.addWidget(top_widget)

        # ---- BOTTOM: Logs & Viewer ----
        bottom_splitter = QSplitter(Qt.Horizontal)
        log_box = QGroupBox("Log")
        lbl_l = QVBoxLayout(log_box)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(10_000)
        lbl_l.addWidget(self.log_edit)

        log_controls = QHBoxLayout()
        b_save_log = QPushButton("Save Log…")
        b_save_log.clicked.connect(self.save_log_to_file)
        b_clear_log = QPushButton("Clear Log")
        b_clear_log.clicked.connect(self.log_edit.clear)
        log_controls.addStretch(1)
        log_controls.addWidget(b_save_log)
        log_controls.addWidget(b_clear_log)
        lbl_l.addLayout(log_controls)

        bottom_splitter.addWidget(log_box)

        viewer_box = QGroupBox("Dataset Viewer")
        vbl = QVBoxLayout(viewer_box)
        vh = QHBoxLayout()
        self.viewer_combo = QComboBox()
        self.viewer_combo.currentIndexChanged.connect(self.load_viewer_items)
        b_open_dir = QPushButton("Open Folder")
        b_open_dir.clicked.connect(self.open_dataset_folder)
        b_export_csv = QPushButton("Export CSV")
        b_export_csv.clicked.connect(self.export_to_csv)
        b_export_json = QPushButton("Export JSON")
        b_export_json.clicked.connect(self.export_to_json_array)
        b_export_sharegpt = QPushButton("Export ShareGPT")
        b_export_sharegpt.clicked.connect(self.export_to_sharegpt)
        b_export_alpaca = QPushButton("Export Alpaca")
        b_export_alpaca.clicked.connect(self.export_to_alpaca)
        vh.addWidget(QLabel("Dataset:"))
        vh.addWidget(self.viewer_combo, 1)
        vh.addWidget(b_export_csv)
        vh.addWidget(b_export_json)
        vh.addWidget(b_export_sharegpt)
        vh.addWidget(b_export_alpaca)
        vh.addWidget(b_open_dir)
        vbl.addLayout(vh)

        # Filter box
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.viewer_filter = QLineEdit()
        self.viewer_filter.setPlaceholderText("Type to filter items by content...")
        self.viewer_filter.textChanged.connect(self._apply_viewer_filter)
        filter_row.addWidget(self.viewer_filter, 1)
        
        b_copy_item = QPushButton("Copy JSON")
        b_copy_item.clicked.connect(self.copy_viewer_item)
        filter_row.addWidget(b_copy_item)
        
        self.stats_label = QLabel("0 items")
        filter_row.addWidget(self.stats_label)
        vbl.addLayout(filter_row)

        vh2 = QSplitter(Qt.Horizontal)
        self.viewer_list = QListWidget()
        self.viewer_list.currentItemChanged.connect(self.display_viewer_item)
        self.viewer_text = QPlainTextEdit()
        self.viewer_text.setReadOnly(True)
        vh2.addWidget(self.viewer_list)
        vh2.addWidget(self.viewer_text)
        vh2.setSizes([220, 420])
        vbl.addWidget(vh2, 1)

        bottom_splitter.addWidget(viewer_box)
        bottom_splitter.setSizes([500, 720])
        self._main_splitter.addWidget(bottom_splitter)
        self._main_splitter.setSizes([600, 340])

    @staticmethod
    def _row(*widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets: h.addWidget(x)
        return w

    # ----------------------------------------------------------------------
    #  Settings / window state
    # ----------------------------------------------------------------------
    def _load_settings(self):
        self.backend_combo.setCurrentText(self.settings.value("backend", "local-llama-cpp", type=str))
        self.temp_spin.setValue(self.settings.value("temp", 0.8, type=float))
        self.max_tokens_spin.setValue(self.settings.value("max_tokens", 2048, type=int))
        self.timeout_spin.setValue(self.settings.value("timeout", 600, type=int))
        self.url_edit.setText(self.settings.value("ollama_url", "http://localhost:11434", type=str))
        self.url_edit_oai.setText(self.settings.value("oai_url", "http://localhost:1234", type=str))
        self.api_key_edit_oai.setText(self.settings.value("oai_api_key", "", type=str))
        self.model_path_edit.setText(self.settings.value("gguf_path", "", type=str))
        self.n_ctx_spin.setValue(self.settings.value("n_ctx", 4096, type=int))
        self.n_gpu_spin.setValue(self.settings.value("n_gpu", 0, type=int))
        self.n_threads_spin.setValue(self.settings.value("n_threads", 8, type=int))
        self.base_dir_edit.setText(self.settings.value("base_dir", str(Path.cwd() / "datasets"), type=str))
        self.batch_spin.setValue(self.settings.value("batch_size", 5, type=int))

    def _restore_window_state(self):
        geo = self.settings.value("geometry")
        if geo: self.restoreGeometry(geo)
        state = self.settings.value("window_state")
        if state: self.restoreState(state)
        ts = self.settings.value("top_splitter")
        if ts: self._top_splitter.restoreState(ts)
        bs = self.settings.value("main_splitter")
        if bs: self._main_splitter.restoreState(bs)

    def closeEvent(self, e):
        self.settings.setValue("backend", self.backend_combo.currentText())
        self.settings.setValue("temp", self.temp_spin.value())
        self.settings.setValue("max_tokens", self.max_tokens_spin.value())
        self.settings.setValue("timeout", self.timeout_spin.value())
        self.settings.setValue("ollama_url", self.url_edit.text())
        self.settings.setValue("oai_url", self.url_edit_oai.text())
        self.settings.setValue("oai_api_key", self.api_key_edit_oai.text())
        self.settings.setValue("gguf_path", self.model_path_edit.text())
        self.settings.setValue("n_ctx", self.n_ctx_spin.value())
        self.settings.setValue("n_gpu", self.n_gpu_spin.value())
        self.settings.setValue("n_threads", self.n_threads_spin.value())
        self.settings.setValue("base_dir", self.base_dir_edit.text())
        self.settings.setValue("batch_size", self.batch_spin.value())
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("top_splitter", self._top_splitter.saveState())
        self.settings.setValue("main_splitter", self._main_splitter.saveState())
        if self.worker:
            self.worker.stop()
            if self.thread:
                self.thread.quit()
                self.thread.wait(3000)
        super().closeEvent(e)

    # ----------------------------------------------------------------------
    #  Fields table
    # ----------------------------------------------------------------------
    def _load_default_fields(self):
        if self.fields_table.rowCount() == 0:
            self.add_field_row("title", "string", "A short descriptive title")
            self.add_field_row("content", "string", "The main body of the entry")
            self.add_field_row("category", "string", "A category label")
            self.add_field_row("tags", "list", "A list of relevant tags")
            self.instruction_edit.setPlainText("Generate diverse, realistic sample entries.")
            self.examples_edit.setPlainText("")
            self.custom_sys_edit.setPlainText("")

    def add_field_row(self, name="", type_="string", desc=""):
        r = self.fields_table.rowCount()
        self.fields_table.insertRow(r)
        self.fields_table.setItem(r, 0, QTableWidgetItem(name))
        cb = QComboBox()
        cb.addItems(FIELD_TYPES)
        cb.setCurrentText(type_)
        self.fields_table.setCellWidget(r, 1, cb)
        self.fields_table.setItem(r, 2, QTableWidgetItem(desc))
        rm = QPushButton("✕")
        rm.setFixedWidth(28)
        rm.clicked.connect(lambda _=False, b=rm: self._remove_row_by_button(b))
        self.fields_table.setCellWidget(r, 3, rm)

    def _remove_row_by_button(self, button):
        for r in range(self.fields_table.rowCount()):
            if self.fields_table.cellWidget(r, 3) is button:
                self.fields_table.removeRow(r)
                return

    def remove_selected_field(self):
        rows = sorted({i.row() for i in self.fields_table.selectedIndexes()}, reverse=True)
        for r in rows: self.fields_table.removeRow(r)

    def clear_fields(self): self.fields_table.setRowCount(0)

    def get_fields(self):
        fields = []
        for r in range(self.fields_table.rowCount()):
            name_item = self.fields_table.item(r, 0)
            desc_item = self.fields_table.item(r, 2)
            cb = self.fields_table.cellWidget(r, 1)
            name = name_item.text().strip() if name_item else ""
            if not name: continue
            fields.append({"name": name, "type": cb.currentText() if cb else "string",
                          "description": desc_item.text().strip() if desc_item else ""})
        return fields

    def get_json_schema(self, batch=False):
        type_map = {"string": "string", "integer": "integer", "float": "number",
                    "boolean": "boolean", "list": "array", "object": "object"}
        properties, required = {}, []
        for f in self.get_fields():
            t = type_map.get(f["type"], "string")
            prop = {"type": t, "description": f["description"]}
            if t == "array": prop["items"] = {"type": "string"}
            properties[f["name"]] = prop
            required.append(f["name"])

        allow_extra = self.allow_extra_props_check.isChecked() if hasattr(self, 'allow_extra_props_check') else False
        item_schema = {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": allow_extra}
        if batch:
            return {"type": "array", "items": item_schema,
                    "minItems": self.batch_spin.value(), "maxItems": self.batch_spin.value()}
        return item_schema

    # ----------------------------------------------------------------------
    #  Backend switching / model loading
    # ----------------------------------------------------------------------
    def on_backend_changed(self, text):
        if text == "local-llama-cpp": self.backend_stack.setCurrentIndex(0)
        elif text == "ollama": self.backend_stack.setCurrentIndex(1)
        else: self.backend_stack.setCurrentIndex(2)

    def browse_gguf(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Models (*.gguf)")
        if p: self.model_path_edit.setText(p)

    def refresh_models(self):
        if requests is None: return QMessageBox.critical(self, "Missing dependency", "pip install requests")
        backend = self.backend_combo.currentText()
        url = self.url_edit.text().strip() if backend == "ollama" else self.url_edit_oai.text().strip()
        api_key = self.api_key_edit_oai.text().strip() if backend != "ollama" else ""
        self.test_label.setText("Fetching models…")
        QGuiApplication.processEvents()
        models = list_models(backend, url, api_key)
        if not models:
            return self.test_label.setText(f"No models found at {url}. Server may not be running.")
        combo = self.model_combo if backend == "ollama" else self.model_combo_oai
        current = combo.currentText()
        combo.clear(); combo.addItems(models)
        if current in models: combo.setCurrentText(current)
        self.test_label.setText(f"Found {len(models)} model(s).")

    def test_connection(self):
        backend = self.backend_combo.currentText()
        if backend == "local-llama-cpp":
            path = self.model_path_edit.text().strip()
            if not path or not os.path.exists(path):
                return self.test_label.setText("Error: Please select a valid .gguf file path.")
            try:
                from llama_cpp import Llama
            except ImportError:
                return self.test_label.setText("Error: 'llama-cpp-python' not installed. Run: pip install llama-cpp-python")
            
            # Fast GGUF Validation via Magic Bytes instead of full load
            try:
                with open(path, "rb") as f:
                    magic = f.read(4)
                    if magic == b"GGUF":
                        self.test_label.setText("Success! Valid GGUF file detected. (Model loads on generation)")
                    else:
                        self.test_label.setText("Error: File exists but is not a valid GGUF file.")
            except Exception as e:
                self.test_label.setText(f"Error reading file: {e}")
        else:
            if requests is None: return QMessageBox.critical(self, "Missing dependency", "pip install requests")
            url = self.url_edit.text().strip() if backend == "ollama" else self.url_edit_oai.text().strip()
            api_key = self.api_key_edit_oai.text().strip() if backend != "ollama" else ""
            models = list_models(backend, url, api_key)
            if models:
                self.test_label.setText(f"OK – connection works. {len(models)} model(s) available.")
            else:
                self.test_label.setText(f"Failed: could not list models at {url}. Check URL and server status.")

    # ----------------------------------------------------------------------
    #  Output dir & wildcards
    # ----------------------------------------------------------------------
    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output base directory", self.base_dir_edit.text())
        if d:
            self.base_dir_edit.setText(d)
            self._update_viewer_datasets()
            self.refresh_wildcard_list()

    def get_wildcard_dir(self):
        wc_dir = Path(self.base_dir_edit.text()) / "wildcards"
        wc_dir.mkdir(parents=True, exist_ok=True)
        return wc_dir

    def _setup_wildcard_watcher(self):
        self._wc_watcher = QFileSystemWatcher()
        self._wc_watcher.directoryChanged.connect(lambda _=None: QTimer.singleShot(200, self.refresh_wildcard_list))
        QTimer.singleShot(300, lambda: self._wc_watcher.addPath(str(self.get_wildcard_dir())))

    def refresh_wildcard_list(self):
        self.wc_list.clear()
        try:
            wc_dir = self.get_wildcard_dir()
            for f in sorted(wc_dir.glob("*.txt")):
                self.wc_list.addItem(f.stem)
            if hasattr(self, '_wc_watcher'):
                if str(wc_dir) not in self._wc_watcher.directories():
                    self._wc_watcher.addPath(str(wc_dir))
        except Exception:
            pass

    def load_wildcard_to_editor(self, current, previous):
        if not current:
            self.wc_editor.clear()
            return
        wc_name = current.text()
        wc_file = self.get_wildcard_dir() / f"{wc_name}.txt"
        if wc_file.exists():
            try:
                with open(wc_file, "r", encoding="utf-8") as f:
                    self.wc_editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def save_wildcard(self):
        if not self.wc_list.currentItem():
            QMessageBox.warning(self, "No Selection", "Please select a wildcard file from the list or create a new one.")
            return
        wc_name = self.wc_list.currentItem().text()
        wc_file = self.get_wildcard_dir() / f"{wc_name}.txt"
        try:
            with open(wc_file, "w", encoding="utf-8") as f:
                f.write(self.wc_editor.toPlainText())
            self.status_label.setText(f"Wildcard '{wc_name}' saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def new_wildcard(self):
        name, ok = QInputDialog.getText(self, "New Wildcard", "Enter wildcard name (no spaces):")
        if not ok or not name.strip(): return
        name = name.strip().replace(" ", "_")
        wc_file = self.get_wildcard_dir() / f"{name}.txt"
        if wc_file.exists():
            QMessageBox.warning(self, "Exists", "A wildcard with this name already exists.")
            return
        wc_file.touch()
        self.refresh_wildcard_list()
        items = self.wc_list.findItems(name, Qt.MatchExactly)
        if items: self.wc_list.setCurrentItem(items[0])

    def delete_wildcard(self):
        if not self.wc_list.currentItem(): return
        wc_name = self.wc_list.currentItem().text()
        ret = QMessageBox.question(self, "Delete", f"Delete wildcard '{wc_name}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            wc_file = self.get_wildcard_dir() / f"{wc_name}.txt"
            if wc_file.exists(): wc_file.unlink()
            self.refresh_wildcard_list()

    # ----------------------------------------------------------------------
    #  Format save / load
    # ----------------------------------------------------------------------
    def save_format(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Format", "format.json", "JSON (*.json)")
        if not path: return
        data = {
            "fields": self.get_fields(),
            "instruction": self.instruction_edit.toPlainText(),
            "examples": self.examples_edit.toPlainText(),
            "custom_system_prompt": self.custom_sys_edit.toPlainText()
        }
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2, ensure_ascii=False)
            self.log(f"Saved format to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def load_format(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Format", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as fp: data = json.load(fp)
            self.fields_table.setRowCount(0)
            for f in data.get("fields", []):
                self.add_field_row(f.get("name", ""), f.get("type", "string"), f.get("description", ""))
            self.instruction_edit.setPlainText(data.get("instruction", ""))
            self.examples_edit.setPlainText(data.get("examples", ""))
            self.custom_sys_edit.setPlainText(data.get("custom_system_prompt", ""))
            self.log(f"Loaded format from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def preview_prompt(self):
        is_batch = self.batch_spin.value() > 1
        schema = self.get_json_schema(batch=is_batch)
        w = GenerationWorker(
            self.get_fields(), self.instruction_edit.toPlainText(),
            self.examples_edit.toPlainText(),
            self.custom_sys_edit.toPlainText(), 1, self.batch_spin.value(),
            {}, "", "", 0, False, self.get_wildcard_dir(), False, (80, 10, 10)
        )
        w.load_wildcards()
        system_prompt = w.build_system_prompt(schema)
        dlg = PromptPreviewDialog(system_prompt, schema, self)
        dlg.exec()

    def log(self, msg):
        self.log_edit.appendPlainText(msg)
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def save_log_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "generation_log.txt", "Text Files (*.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.log_edit.toPlainText())
                self.status_label.setText(f"Log saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # ----------------------------------------------------------------------
    #  Viewer
    # ----------------------------------------------------------------------
    def _update_viewer_datasets(self):
        base_dir = Path(self.base_dir_edit.text())
        self.viewer_combo.blockSignals(True)
        current = self.viewer_combo.currentText()
        self.viewer_combo.clear()
        if base_dir.exists():
            dirs = sorted([d.name for d in base_dir.iterdir()
                          if d.is_dir() and (d / "items.jsonl").exists()])
            self.viewer_combo.addItems(dirs)
            if current in dirs:
                self.viewer_combo.setCurrentText(current)
        self.viewer_combo.blockSignals(False)
        if self.viewer_combo.count() > 0:
            self.load_viewer_items(self.viewer_combo.currentIndex())
        else:
            self.viewer_list.clear()
            self.viewer_text.clear()
            self._update_dataset_stats()

    def load_viewer_items(self, idx):
        self.viewer_list.clear()
        self.viewer_text.clear()
        self._viewer_items_cache = []
        if idx < 0: 
            self._update_dataset_stats()
            return
        ds_name = self.viewer_combo.currentText()
        if not ds_name:
            self._update_dataset_stats()
            return
        jsonl_path = Path(self.base_dir_edit.text()) / ds_name / "items.jsonl"
        if not jsonl_path.exists():
            self._update_dataset_stats()
            return
        items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    items.append((i+1, obj))
                except Exception:
                    pass
        self._viewer_items_cache = items
        self._apply_viewer_filter(self.viewer_filter.text())

    def _apply_viewer_filter(self, text):
        self.viewer_list.clear()
        text = text.strip().lower()
        for idx, obj in getattr(self, '_viewer_items_cache', []):
            blob = json.dumps(obj, ensure_ascii=False).lower()
            if not text or text in blob:
                li = QListWidgetItem(f"Item {idx}")
                li.setData(Qt.UserRole, obj)
                self.viewer_list.addItem(li)
        self._update_dataset_stats()

    def _update_dataset_stats(self):
        items = [obj for _, obj in getattr(self, '_viewer_items_cache', [])]
        n = len(items)
        if n == 0:
            self.stats_label.setText("0 items")
            return
        # Gather per-field emptiness
        field_stats = {}
        for f in self.get_fields():
            name = f["name"]
            empties = sum(1 for it in items if not it.get(name))
            field_stats[name] = empties
        tip_lines = [f"Total: {n} items"]
        for name, emp in field_stats.items():
            tip_lines.append(f"  {name}: {emp} empty ({100*emp//n}%)")
        self.stats_label.setText(f"{n} items")
        self.stats_label.setToolTip("\n".join(tip_lines))

    def display_viewer_item(self, current, previous):
        if not current: return self.viewer_text.clear()
        obj = current.data(Qt.UserRole)
        self.viewer_text.setPlainText(json.dumps(obj, indent=2, ensure_ascii=False))

    def copy_viewer_item(self):
        text = self.viewer_text.toPlainText()
        if text:
            QGuiApplication.clipboard().setText(text)
            self.status_label.setText("Item JSON copied to clipboard.")

    def open_dataset_folder(self):
        ds_name = self.viewer_combo.currentText()
        if not ds_name: return
        ds_path = Path(self.base_dir_edit.text()) / ds_name
        if ds_path.exists():
            if sys.platform == "win32": os.startfile(ds_path)
            elif sys.platform == "darwin": os.system(f'open "{ds_path}"')
            else: os.system(f'xdg-open "{ds_path}"')

    # ----------------------------------------------------------------------
    #  Exports
    # ----------------------------------------------------------------------
    def _load_viewer_jsonl(self):
        ds_name = self.viewer_combo.currentText()
        if not ds_name: return None
        jsonl_path = Path(self.base_dir_edit.text()) / ds_name / "items.jsonl"
        if not jsonl_path.exists(): return None
        items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): items.append(json.loads(line))
        return ds_name, items

    def export_to_csv(self):
        result = self._load_viewer_jsonl()
        if not result: return
        ds_name, items = result
        if not items: return QMessageBox.information(self, "Empty", "No items to export.")
        csv_path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", f"{ds_name}.csv", "CSV Files (*.csv)")
        if not csv_path: return
        try:
            flat_items = []
            headers = set()
            for item in items:
                flat = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in item.items()}
                flat_items.append(flat)
                headers.update(flat.keys())
            headers = sorted(list(headers))
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(flat_items)
            QMessageBox.information(self, "Exported", f"Exported {len(flat_items)} rows to CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_to_json_array(self):
        result = self._load_viewer_jsonl()
        if not result: return
        ds_name, items = result
        if not items: return QMessageBox.information(self, "Empty", "No items to export.")
        json_path, _ = QFileDialog.getSaveFileName(self, "Export to JSON Array", f"{ds_name}.json", "JSON Files (*.json)")
        if not json_path: return
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Exported", f"Exported {len(items)} items to JSON array.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_to_sharegpt(self):
        result = self._load_viewer_jsonl()
        if not result: return
        ds_name, items = result
        if not items: return QMessageBox.information(self, "Empty", "No items to export.")

        json_path, _ = QFileDialog.getSaveFileName(self, "Export ShareGPT", f"{ds_name}_sharegpt.json", "JSON Files (*.json)")
        if not json_path: return
        try:
            convs = []
            for it in items:
                if "conversations" in it and isinstance(it["conversations"], list):
                    conv = [{"from": c.get("from", "human"), "value": c.get("value", "")}
                            for c in it["conversations"]]
                else:
                    fields = list(it.items())
                    if not fields: continue
                    conv = [{"from": "human", "value": str(fields[0][1])},
                            {"from": "gpt",   "value": "\n\n".join(str(v) for _, v in fields[1:])}]
                convs.append({"conversations": conv})
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(convs, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Exported", f"Exported {len(convs)} ShareGPT conversations.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_to_alpaca(self):
        result = self._load_viewer_jsonl()
        if not result: return
        ds_name, items = result
        if not items: return QMessageBox.information(self, "Empty", "No items to export.")

        json_path, _ = QFileDialog.getSaveFileName(self, "Export Alpaca", f"{ds_name}_alpaca.json", "JSON Files (*.json)")
        if not json_path: return
        try:
            out = []
            for it in items:
                instr = it.get("instruction", "")
                inp = it.get("input", "")
                outp = it.get("output", it.get("content", it.get("response", "")))
                if not instr:
                    for k, v in it.items():
                        if isinstance(v, str) and v.strip():
                            instr = v
                            outp = "\n".join(str(x) for j, x in it.items() if j != k)
                            break
                out.append({"instruction": instr, "input": inp, "output": outp})
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Exported", f"Exported {len(out)} Alpaca-style examples.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # ----------------------------------------------------------------------
    #  Generation control
    # ----------------------------------------------------------------------
    def sanitize_filename(self, name):
        return re.sub(r'[\\/*?:"<>|]', "_", name).strip()

    def start_generation(self):
        if self.worker: return
        fields = self.get_fields()
        if not fields and not self.custom_sys_edit.toPlainText().strip():
            return QMessageBox.warning(self, "No fields",
                "Add at least one field or provide a Custom System Prompt.")

        backend = self.backend_combo.currentText()
        client_config = {
            "backend": backend,
            "temperature": self.temp_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "timeout": self.timeout_spin.value(),
            "api_key": self.api_key_edit_oai.text().strip() if backend == "openai-compatible" else ""
        }

        if backend == "local-llama-cpp":
            path = self.model_path_edit.text().strip()
            if not path or not os.path.exists(path):
                return QMessageBox.warning(self, "No Model", "Please select a valid .gguf model file.")
            client_config["model_path"] = path
            client_config["n_ctx"] = self.n_ctx_spin.value()
            client_config["n_gpu_layers"] = self.n_gpu_spin.value()
            client_config["n_threads"] = self.n_threads_spin.value()
        else:
            if requests is None:
                return QMessageBox.critical(self, "Missing dependency", "pip install requests")
            if backend == "ollama":
                client_config["base_url"] = self.url_edit.text().strip()
                client_config["model"] = self.model_combo.currentText().strip()
            else:
                client_config["base_url"] = self.url_edit_oai.text().strip()
                client_config["model"] = self.model_combo_oai.currentText().strip()
            if not client_config.get("model"):
                return QMessageBox.warning(self, "No model", "Select or type a model name.")
            if not client_config.get("base_url"):
                return QMessageBox.warning(self, "No URL", "Enter the base URL.")

        output_dir = self.base_dir_edit.text().strip()
        if not output_dir: return QMessageBox.warning(self, "No output dir", "Choose an output directory.")
        
        dataset_name = self.sanitize_filename(self.dataset_name_edit.text().strip())
        if not dataset_name: return QMessageBox.warning(self, "No name", "Enter a dataset name.")
        self.dataset_name_edit.setText(dataset_name)  # Reflect sanitized name

        split_ratios = (100, 0, 0)
        if self.split_btn_check.isChecked():
            t_sum = self.train_spin.value() + self.test_spin.value() + self.val_spin.value()
            if t_sum != 100:
                return QMessageBox.warning(self, "Invalid Split",
                    f"Train + Test + Val must equal 100%. Currently: {t_sum}%")
            split_ratios = (self.train_spin.value(), self.test_spin.value(), self.val_spin.value())

        # Pre-generation wildcard validation
        wc_dir = Path(output_dir) / "wildcards"
        wc_files = [f.stem for f in wc_dir.glob("*.txt")] if wc_dir.exists() else []
        text_to_check = (self.instruction_edit.toPlainText() + " "
                        + self.examples_edit.toPlainText() + " "
                        + self.custom_sys_edit.toPlainText() + " "
                        + " ".join([f["description"] for f in fields]))
        missing_wcs = []
        for match in WILDCARD_REGEX.finditer(text_to_check):
            wc_name = match.group(1)
            if wc_name not in wc_files and wc_name not in missing_wcs:
                missing_wcs.append(wc_name)
        if missing_wcs:
            ret = QMessageBox.warning(self, "Missing Wildcards",
                f"The following wildcards were used but not found in the 'wildcards' folder:\n"
                f"{', '.join(missing_wcs)}\n\nGenerate anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret == QMessageBox.No: return

        ds_path = Path(output_dir) / dataset_name
        if ds_path.exists() and not self.append_check.isChecked():
            ret = QMessageBox.question(self, "Dataset exists",
                f"Dataset folder '{ds_path}' already exists.\nAppend to it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret != QMessageBox.Yes: return
            self.append_check.setChecked(True)

        n_items = self.count_spin.value()
        instruction = self.instruction_edit.toPlainText()
        examples = self.examples_edit.toPlainText()
        custom_sys = self.custom_sys_edit.toPlainText()
        wc_dir_path = Path(output_dir) / "wildcards"

        self.thread = QThread()
        self.worker = GenerationWorker(
            fields, instruction, examples, custom_sys, n_items, self.batch_spin.value(),
            client_config, output_dir, dataset_name, self.delay_spin.value(),
            self.append_check.isChecked(), wc_dir_path, self.dedup_check.isChecked(),
            split_ratios,
            allow_extra_properties=self.allow_extra_props_check.isChecked(),
            continue_on_error=self.continue_on_error_check.isChecked(),
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.item_generated.connect(self._on_item)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)

        self.gen_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setRange(0, n_items)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Generating 0 / {n_items} …")
        self.thread.start()

    def stop_generation(self):
        if self.worker:
            self.worker.stop()
            self.log("Stop requested…")

    def _on_progress(self, done, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Generated {done} / {total}")

    def _on_item(self, idx, obj):
        pass

    def _on_finished(self, ok, msg):
        self.log(("✓ " if ok else "✗ ") + msg)
        self.gen_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(msg)
        if self.thread:
            self.thread.wait()
            self.thread.deleteLater()
        if self.worker:
            self.worker.deleteLater()
        self.worker = None
        self.thread = None
        self._update_viewer_datasets()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 10))
    app.setStyleSheet(DARK_STYLESHEET)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()