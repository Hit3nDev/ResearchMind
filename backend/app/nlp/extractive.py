"""Classical (non-LLM) summarisation and information extraction.

* Extractive summarisation: TF-IDF sentence scoring with a position prior.
* Information extraction: regular expressions + gazetteers for model, dataset
  and metric mentions; cue-phrase sentence selection for limitations.
"""
import math
import re
from collections import Counter

from app.models.schemas import ExtractedInfo
from app.nlp.preprocessing import preprocess, sentence_tokenize

MODEL_PATTERN = re.compile(
    r"\b(YOLO(?:v\d+|-?NAS)?|Faster R-CNN|Mask R-CNN|R-CNN|ResNet-?\d*|VGG-?\d*|EfficientNet(?:-?B\d)?|"
    r"MobileNet(?:-?V\d)?|DenseNet-?\d*|Inception(?:-?v\d)?|Xception|AlexNet|ConvNeXt|U-Net|Swin Transformer|"
    r"Vision Transformer|ViT|DeiT|DETR|CLIP|BERT|RoBERTa|DistilBERT|DeBERTa|XLNet|ALBERT|GPT-?\d*|T5|BART|"
    r"LLaMA-?\d*|Llama-?\d*|Mistral|LSTM|BiLSTM|GRU|CNN|RNN|Transformer|GNN|GCN|GAT|GAN|VAE|"
    r"SVM|Random Forest|XGBoost|LightGBM|k-NN|KNN|Naive Bayes|Logistic Regression|Decision Tree|"
    r"Word2Vec|GloVe|fastText|ELMo)\b")
GENERIC_MODELS = {"CNN", "RNN", "Transformer", "GNN"}
KNOWN_DATASETS = re.compile(
    r"\b(ImageNet|CIFAR-?10{1,2}|MNIST|Fashion-MNIST|MS ?COCO|COCO|Pascal VOC|PlantVillage|PlantDoc|"
    r"SQuAD|GLUE|SuperGLUE|IMDB|SST-?2|CoNLL-?\d*|WMT\d*|LibriSpeech|Kinetics|UCF101|Cityscapes|KITTI|"
    r"MIMIC-?I*V?|ChestX-?ray14|CheXpert|ISIC|BraTS|Penn Treebank|WikiText-?\d*|HumanEval|MMLU|GSM8K)\b", re.I)
DATASET_PATTERN = re.compile(r"\b((?:[A-Z][\w-]*\s){0,2}[A-Z][\w-]*)\s+(?:dataset|corpus|benchmark)\b")
DATASET_BLACKLIST = {"the", "this", "our", "a", "an", "these", "public", "new", "large", "each", "same", "test", "training"}
LIMIT_CUES = re.compile(r"\b(limitation|limited|drawback|shortcoming|future work|fails? to|cannot|small (?:dataset|sample)|"
                        r"only (?:on|one|a single)|not (?:yet|evaluated|tested)|restricted to|challenging)\b", re.I)


def _metric(text: str, name: str) -> str | None:
    pats = [rf"{name}\s*(?:of|is|was|=|:|reached|reaching|achieved|achieves|achieving|up to|about|around|\s)*\s*(\d{{1,3}}(?:\.\d+)?)\s?%",
            rf"(\d{{1,3}}(?:\.\d+)?)\s?%\s*(?:\w+\s){{0,2}}{name}"]
    vals = []
    for p in pats:
        for m in re.finditer(p, text, re.I):
            v = float(m.group(1))
            if 1 < v <= 100:
                vals.append(v)
    return f"{max(vals):g}%" if vals else None


def extract_info(abstract: str, full_text: str | None) -> ExtractedInfo:
    text = f"{abstract}\n{full_text or ''}"
    models = Counter(m.group(1) for m in MODEL_PATTERN.finditer(text))
    specific = [m for m, _ in models.most_common() if m not in GENERIC_MODELS]
    model = specific[0] if specific else (models.most_common(1)[0][0] if models else None)

    found = [m.group(1) for m in KNOWN_DATASETS.finditer(text)] or \
        [m.group(1).strip() for m in DATASET_PATTERN.finditer(text) if m.group(1).split()[0].lower() not in DATASET_BLACKLIST]
    datasets = Counter(d.lower() for d in found)
    dataset = None
    if datasets:
        top = datasets.most_common(1)[0][0]
        variants = Counter(d for d in found if d.lower() == top)
        # display the most common spelling that has capitals ("WMT", not "wmt")
        dataset = max(variants, key=lambda v: (v != v.lower(), variants[v]))

    limitations = []
    for s in sentence_tokenize(full_text or abstract):
        if LIMIT_CUES.search(s) and 6 < len(s.split()) < 45:
            limitations.append(re.sub(r"\s+", " ", s).strip())
        if len(limitations) == 3:
            break
    return ExtractedInfo(model=model, dataset=dataset, accuracy=_metric(text, "accuracy"),
                         precision=_metric(text, "precision"), recall=_metric(text, "recall"),
                         limitations=limitations)


def summarize(text: str, n_sentences: int = 1) -> str:
    """Score each sentence by the sum of its terms' TF-IDF weights (sentences = documents)."""
    sents = [s for s in sentence_tokenize(text) if 5 <= len(s.split()) <= 60]
    if not sents:
        return text[:200]
    tokenized = [preprocess(s) for s in sents]
    df = Counter(t for toks in tokenized for t in set(toks))
    tf_doc = Counter(t for toks in tokenized for t in toks)
    N = len(sents)
    scores = []
    for i, toks in enumerate(tokenized):
        w = sum(tf_doc[t] * math.log((N + 1) / (df[t] + 0.5)) for t in set(toks)) / (len(toks) + 1) ** 0.5
        w *= 1.3 if i == 0 else (1.15 if re.search(r"\b(we propose|this paper|we present|results show)\b", sents[i], re.I) else 1)
        scores.append(w)
    best = sorted(range(N), key=lambda i: -scores[i])[:n_sentences]
    return " ".join(sents[i] for i in sorted(best))
