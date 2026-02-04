import os
import json
import csv
import re
from collections import Counter
from PIL import Image
from tqdm import tqdm

from ..base_dataset import BaseDataset

HEADERS = ["【影像所见】", "【印象】", "【BI-RADS】", "【建议】"]

DEFAULT_PROMPT = (
    "请根据这张2D乳腺超声图像生成中文诊断/病理风格报告，使用结构化输出：\n"
    "【影像所见】\n"
    "【印象】\n"
    "【BI-RADS】\n"
    "【建议】\n"
    "如有不确定之处，请明确说明不确定。"
)


def _read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_reports(dataset_path):
    reports = {}
    jsonl_path = os.path.join(dataset_path, "reports.jsonl")
    json_path = os.path.join(dataset_path, "reports.json")
    csv_path = os.path.join(dataset_path, os.environ.get("BREASTUS_REPORTS_CSV", "abus_b_g.csv"))
    if os.path.exists(jsonl_path):
        items = _read_jsonl(jsonl_path)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "report": v} for k, v in data.items()]
        else:
            items = data
    elif os.path.exists(csv_path):
        items = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                patient = str(row[1]).strip()
                report = str(row[2]).strip()
                if not patient or not report:
                    continue
                if patient in ("超声号", "patient_id", "patient"):
                    continue
                items.append({"patient": patient, "report": report})
    else:
        return reports

    for item in items:
        key = item.get("id") or item.get("patient")
        if key is None:
            continue
        reports[str(key)] = item
    return reports


def _normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _char_f1(pred, gold):
    pred = _normalize_text(pred)
    gold = _normalize_text(gold)
    if not pred or not gold:
        return 0.0
    pred_chars = [c for c in pred if not c.isspace()]
    gold_chars = [c for c in gold if not c.isspace()]
    pred_counter = Counter(pred_chars)
    gold_counter = Counter(gold_chars)
    common = pred_counter & gold_counter
    common_count = sum(common.values())
    if common_count == 0:
        return 0.0
    precision = common_count / max(1, len(pred_chars))
    recall = common_count / max(1, len(gold_chars))
    return 2 * precision * recall / (precision + recall)


def _extract_birads(text):
    items = _extract_birads_all(text)
    return items[0] if items else None


def _extract_birads_all(text):
    if not text:
        return []
    text = text.upper()
    results = []

    # Strong pattern with explicit BI-RADS mention
    for m in re.finditer(r"BI[- ]?RADS[^0-6]*([0-6])\s*([ABC])?", text):
        grade = m.group(1)
        suffix = m.group(2) or ""
        results.append(f"{grade}{suffix}")

    # Generic patterns like "4A" or "3类" (avoid matching clock numbers like 11点)
    for m in re.finditer(r"(?<!\d)([0-6])(?!\d)\s*([ABC])\b", text):
        results.append(f"{m.group(1)}{m.group(2)}")
    for m in re.finditer(r"(?<!\d)([0-6])(?!\d)\s*类", text):
        results.append(f"{m.group(1)}")

    # Deduplicate, keep order
    seen = set()
    uniq = []
    for r in results:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _structure_ok(text):
    return all(h in text for h in HEADERS)


def _uniform_sample(items, k):
    if k <= 0 or k >= len(items):
        return items
    if k == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (k - 1)
    idxs = [int(round(i * step)) for i in range(k)]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def _strip_headers(text):
    if not text:
        return ""
    lines = text.splitlines()
    kept = []
    for line in lines:
        if any(h in line for h in HEADERS):
            continue
        kept.append(line)
    return _normalize_text("\n".join(kept))


def _build_structured_report(raw_text, birads_list=None):
    birads_text = "/".join(birads_list or [])
    raw_text = (raw_text or "").strip()
    return (
        f"{HEADERS[0]}\n\n"
        f"{HEADERS[1]}\n{raw_text}\n"
        f"{HEADERS[2]}\n{birads_text}\n"
        f"{HEADERS[3]}\n"
    )


def _birads_set_metrics(gold_list, pred_list):
    gold_set = set(gold_list or [])
    pred_set = set(pred_list or [])
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact": 1.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact": 0.0}
    inter = gold_set & pred_set
    precision = len(inter) / max(1, len(pred_set))
    recall = len(inter) / max(1, len(gold_set))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = 1.0 if gold_set == pred_set else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "exact": exact}



class BreastUS(BaseDataset):
    def __init__(self, model, dataset_path, output_path, mode="single"):
        self.model = model
        self.output_path = output_path
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.images_dir, self.masks_dir = self._resolve_data_dirs()
        self.mode = mode  # single | labeled | full
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        self.reports = _load_reports(dataset_path)

        self.prompt = os.environ.get("BREASTUS_PROMPT", "").strip()
        if not self.prompt:
            prompt_file = os.environ.get("BREASTUS_PROMPT_FILE", "").strip()
            if prompt_file and os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self.prompt = f.read().strip()
        if not self.prompt:
            self.prompt = DEFAULT_PROMPT

    def _resolve_dataset_path(self, dataset_path):
        if not dataset_path:
            return dataset_path
        path = os.path.abspath(dataset_path)
        if os.path.isdir(os.path.join(path, "images")) or os.path.isdir(os.path.join(path, "trainimg")):
            return path
        parent = os.path.dirname(path)
        if os.path.isdir(os.path.join(parent, "images")) or os.path.isdir(os.path.join(parent, "trainimg")):
            return parent
        return path

    def _resolve_data_dirs(self):
        # Allow custom folder names via env, with sensible fallbacks.
        images_dir = os.environ.get("BREASTUS_IMAGES_DIR", "images")
        masks_dir = os.environ.get("BREASTUS_MASKS_DIR", "masks")

        images_root = os.path.join(self.dataset_path, images_dir)
        masks_root = os.path.join(self.dataset_path, masks_dir)

        if not os.path.isdir(images_root):
            if os.path.isdir(os.path.join(self.dataset_path, "trainimg")):
                images_dir = "trainimg"
            elif os.path.isdir(os.path.join(self.dataset_path, "img")):
                images_dir = "img"

        if not os.path.isdir(masks_root):
            if os.path.isdir(os.path.join(self.dataset_path, "trainlabel")):
                masks_dir = "trainlabel"
            elif os.path.isdir(os.path.join(self.dataset_path, "label")):
                masks_dir = "label"

        return images_dir, masks_dir

    def _get_target_len(self, n):
        user_len = int(os.environ.get("BREASTUS_SEQ_LEN", "0"))
        target = n if user_len <= 0 else min(user_len, n)
        model_cap = int(os.environ.get("max_image_num", "0"))
        if model_cap > 0:
            target = min(target, model_cap)
        return target

    def _resolve_report(self, sample):
        sid = str(sample.get("id", ""))
        patient = str(sample.get("patient", ""))
        report = self.reports.get(sid) or self.reports.get(patient)
        if not report:
            return
        if "report" in report:
            sample["report"] = report["report"]
        else:
            for key in ["findings", "impression", "birads", "recommendation"]:
                if key in report:
                    sample[key] = report[key]

    def _load_index(self, filename):
        path = os.path.join(self.dataset_path, filename)
        if os.path.exists(path):
            return _read_jsonl(path)
        return None

    def _auto_single(self):
        images_root = os.path.join(self.dataset_path, self.images_dir)
        masks_root = os.path.join(self.dataset_path, self.masks_dir)
        items = []
        for patient in sorted(os.listdir(images_root)):
            p_dir = os.path.join(images_root, patient)
            if not os.path.isdir(p_dir):
                continue
            for fname in sorted(os.listdir(p_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                rel_img = os.path.join(self.images_dir, patient, fname)
                sample = {"id": f"{patient}_{fname}", "patient": patient, "image": rel_img}
                rel_mask = os.path.join(self.masks_dir, patient, fname)
                if os.path.exists(os.path.join(self.dataset_path, rel_mask)):
                    sample["mask"] = rel_mask
                self._resolve_report(sample)
                items.append(sample)
        return items

    def _auto_labeled_seq(self):
        masks_root = os.path.join(self.dataset_path, self.masks_dir)
        images_root = os.path.join(self.dataset_path, self.images_dir)
        items = []
        for patient in sorted(os.listdir(masks_root)):
            p_dir = os.path.join(masks_root, patient)
            if not os.path.isdir(p_dir):
                continue
            mask_files = [f for f in sorted(os.listdir(p_dir)) if f.lower().endswith(".png")]
            if not mask_files:
                continue
            rel_images = []
            rel_masks = []
            for fname in mask_files:
                rel_mask = os.path.join(self.masks_dir, patient, fname)
                rel_img = os.path.join(self.images_dir, patient, fname)
                if not os.path.exists(os.path.join(self.dataset_path, rel_img)):
                    continue
                rel_images.append(rel_img)
                rel_masks.append(rel_mask)
            if not rel_images:
                continue
            sample = {"id": f"{patient}_labeled", "patient": patient, "images": rel_images, "masks": rel_masks}
            self._resolve_report(sample)
            items.append(sample)
        return items

    def _auto_full_seq(self):
        images_root = os.path.join(self.dataset_path, self.images_dir)
        items = []
        for patient in sorted(os.listdir(images_root)):
            p_dir = os.path.join(images_root, patient)
            if not os.path.isdir(p_dir):
                continue
            img_files = [f for f in sorted(os.listdir(p_dir)) if f.lower().endswith(".png")]
            if not img_files:
                continue
            rel_images = [os.path.join(self.images_dir, patient, f) for f in img_files]
            sample = {"id": f"{patient}_full", "patient": patient, "images": rel_images}
            self._resolve_report(sample)
            items.append(sample)
        return items

    def load_data(self):
        if self.mode == "single":
            items = self._load_index("index_single.jsonl") or self._auto_single()
        elif self.mode == "labeled":
            items = self._load_index("index_labeled_seq.jsonl") or self._auto_labeled_seq()
        else:
            items = self._load_index("index_full_seq.jsonl") or self._auto_full_seq()

        for idx, sample in tqdm(enumerate(items)):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            sample = self.construct_messages(sample)
            self.samples.append(sample)
        return self.samples

    def construct_messages(self, sample):
        if self.mode == "single":
            img_path = os.path.join(self.dataset_path, sample["image"])
            image = Image.open(img_path)
            sample["messages"] = {"prompt": self.prompt, "image": image}
            return sample

        image_paths = sample["images"]
        target_len = self._get_target_len(len(image_paths))
        image_paths = _uniform_sample(image_paths, target_len)
        images = [Image.open(os.path.join(self.dataset_path, p)) for p in image_paths]
        sample["messages"] = {"prompt": self.prompt, "images": images}
        sample["selected_images"] = image_paths
        return sample

    def cal_metrics(self, out_samples):
        total = len(out_samples)
        if total == 0:
            return {"total metrics": {"total": 0}}, out_samples

        structure_ok = 0
        birads_total = 0
        birads_exact_sum = 0.0
        birads_f1_sum = 0.0
        birads_prec_sum = 0.0
        birads_rec_sum = 0.0
        char_f1_sum = 0.0
        char_f1_count = 0
        content_f1_sum = 0.0
        content_f1_count = 0

        for i, sample in enumerate(out_samples):
            response = sample.get("response", "")
            if _structure_ok(response):
                structure_ok += 1
            sample["structure_ok"] = _structure_ok(response)

            gold_report = sample.get("report")
            findings = sample.get("findings")
            impression = sample.get("impression")
            gold_birads = sample.get("birads")
            recommendation = sample.get("recommendation")

            if not gold_report and any([findings, impression, gold_birads, recommendation]):
                gold_report = (
                    f"{HEADERS[0]}\n{findings or ''}\n"
                    f"{HEADERS[1]}\n{impression or ''}\n"
                    f"{HEADERS[2]}\n{gold_birads or ''}\n"
                    f"{HEADERS[3]}\n{recommendation or ''}"
                )

            if gold_report:
                gold_birads_list = _extract_birads_all(gold_report)
                if not _structure_ok(gold_report):
                    gold_report = _build_structured_report(gold_report, gold_birads_list)
                f1 = _char_f1(response, gold_report)
                char_f1_sum += f1
                char_f1_count += 1
                sample["char_f1"] = f1

                content_f1 = _char_f1(_strip_headers(response), _strip_headers(gold_report))
                content_f1_sum += content_f1
                content_f1_count += 1
                sample["content_char_f1"] = content_f1

                pred_birads_list = _extract_birads_all(response)
                if gold_birads_list:
                    birads_total += 1
                    metrics = _birads_set_metrics(gold_birads_list, pred_birads_list)
                    birads_exact_sum += metrics["exact"]
                    birads_f1_sum += metrics["f1"]
                    birads_prec_sum += metrics["precision"]
                    birads_rec_sum += metrics["recall"]
                    sample["birads_gold_all"] = gold_birads_list
                    sample["birads_pred_all"] = pred_birads_list

        metrics = {
            "total metrics": {
                "total": total,
                "structure_ok": structure_ok,
                "structure_rate": structure_ok / total,
            }
        }
        if char_f1_count > 0:
            metrics["total metrics"]["char_f1"] = char_f1_sum / char_f1_count
        if content_f1_count > 0:
            metrics["total metrics"]["content_char_f1"] = content_f1_sum / content_f1_count
        if birads_total > 0:
            metrics["total metrics"]["birads_exact_acc"] = birads_exact_sum / birads_total
            metrics["total metrics"]["birads_f1"] = birads_f1_sum / birads_total
            metrics["total metrics"]["birads_precision"] = birads_prec_sum / birads_total
            metrics["total metrics"]["birads_recall"] = birads_rec_sum / birads_total

        return metrics, out_samples
