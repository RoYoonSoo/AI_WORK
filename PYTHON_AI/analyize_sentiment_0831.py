import warnings
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import os
import glob
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# pandas에서 openpyxl 관련 경고 무시
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------
# 1. 텍스트 데이터셋 클래스 정의 (✅ 수정된 부분)
# ---------------------------------------------------------
class TextCommentDataset(Dataset):
    def __init__(self, data_folder_path, nli_tokenizer, nli_model, device):
        self.device = device

        file_paths = glob.glob(os.path.join(data_folder_path, '**', '*.*'), recursive=True)
        file_paths = [f for f in file_paths if os.path.isfile(f) and not os.path.basename(f).startswith('.')]

        all_dfs = []
        for p in file_paths:
            try:
                if p.endswith('.csv'):
                    try:
                        all_dfs.append(pd.read_csv(p, header=None, skiprows=1, encoding='utf-8-sig'))
                    except UnicodeDecodeError:
                        try:
                            all_dfs.append(pd.read_csv(p, header=None, skiprows=1, encoding='cp949'))
                        except UnicodeDecodeError:
                            all_dfs.append(pd.read_csv(p, header=None, skiprows=1, encoding='utf-8', errors='replace'))
                elif p.endswith('.xlsx'):
                    all_dfs.append(pd.read_excel(p, header=None, skiprows=1))
            except Exception as e:
                print(f"파일을 읽는 중 오류 발생: {p}, 오류: {e}")

        if not all_dfs:
            combined_df = pd.DataFrame()
        else:
            combined_df = pd.concat(all_dfs, ignore_index=True)

        if combined_df.empty:
            self.comments = []
            self.labels = []
            return

        self.comments = combined_df[0].fillna('').tolist()

        # 🔹 제로샷 NLI 모델로 라벨 생성
        print("\n🔹 제로샷 NLI 모델로 모든 댓글의 라벨을 생성합니다... (시간이 매우 오래 걸릴 수 있습니다)")
        hypotheses = {
            "칭찬": "이 문장은 작품에 대한 칭찬이다.",
            "비판": "이 문장은 작품에 대한 비판이다.",
            "작품내용": "이 문장은 작품과 관련된 내용이다.",
            "논쟁": "이 문장은 댓글여론과 논쟁을 펼치고 있다.",
            "관련없음": "이 문장은 작품에 대해서 이야기하는 것이 아니다",
        }
        try:
            entailment_index = nli_model.config.label2id['entailment']
        except KeyError:
            entailment_index = 0

        predicted_labels = []
        total_comments = len(self.comments)
        for i, comment in enumerate(self.comments):
            if (i + 1) % 10 == 0:
                print(f"  - 라벨링 진행 중: {i + 1} / {total_comments}")

            if not comment.strip():
                predicted_labels.append(-1)  # 유효하지 않은 라벨
                continue

            entailment_scores = {}
            for label_name, hypothesis in hypotheses.items():
                inputs = nli_tokenizer(comment, hypothesis, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = nli_model(**inputs).logits
                    entailment_scores[label_name] = logits[0][entailment_index].item()

            best_label = max(entailment_scores, key=entailment_scores.get)
            label_map = {"칭찬": 0, "비판": 1, "작품내용": 2, "논쟁": 3, "관련없음": 4}
            predicted_labels.append(label_map[best_label])

        self.labels = predicted_labels
        print("✅ 모든 댓글의 라벨 생성이 완료되었습니다.")

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        # ✅ 핵심 수정: 학습에 필요한 '댓글 원본'과 '라벨'만 반환하도록 변경
        return {
            'comment': self.comments[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------------------------------------------------------
# 2. 텍스트 분류 모델 클래스
# ---------------------------------------------------------
class TextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits


# ---------------------------------------------------------
# 3. K-Fold 학습 진행 (✅ 수정된 부분)
# ---------------------------------------------------------
def train_with_kfold(data_folder_path, epochs=5, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 현재 사용 장치: {device} ---")

    # 학습에 사용할 모델과 토크나이저
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 제로샷 라벨링용 모델
    print("\n--- 제로샷 라벨링 모델 로딩 ---")
    nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    print("  - 모델 로딩 완료.")

    # 데이터셋 로딩 및 라벨 생성
    print("\n--- 데이터셋 로딩 및 라벨 생성 시작 ---")
    full_dataset = TextCommentDataset(data_folder_path, nli_tokenizer, nli_model, device)
    if len(full_dataset) == 0:
        print("⚠️ 학습할 데이터가 없습니다.")
        return

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\n===== Fold {fold + 1} / {k} =====")

        train_subset_raw = Subset(full_dataset, train_idx)
        val_subset_raw = Subset(full_dataset, val_idx)

        # -1 라벨을 가진 데이터를 필터링
        train_subset = [s for s in train_subset_raw if s['labels'] != -1]
        val_subset = [s for s in val_subset_raw if s['labels'] != -1]

        if not train_subset or not val_subset:
            print("  - Fold에 학습 또는 검증 데이터가 없어 건너뜁니다.")
            continue

        train_dl = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_dl = DataLoader(val_subset, batch_size=16)

        model = TextClassifier(MODEL_NAME, num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=5e-5)

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_dl:
                # ✅ 핵심 수정: 이제 batch['comment']에 접근 가능
                # 학습용 토크나이저(klue/bert-base)로 배치마다 토큰화
                batch_inputs = tokenizer(batch['comment'], return_tensors='pt', padding=True, truncation=True,
                                         max_length=128).to(device)

                logits = model(batch_inputs['input_ids'], batch_inputs['attention_mask'])
                loss = criterion(logits, batch['labels'].to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()

            # 검증
            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for batch in val_dl:
                    # ✅ 핵심 수정: 검증 데이터도 동일하게 처리
                    batch_inputs = tokenizer(batch['comment'], return_tensors='pt', padding=True, truncation=True,
                                             max_length=128).to(device)
                    logits = model(batch_inputs['input_ids'], batch_inputs['attention_mask'])
                    pred = torch.argmax(logits, dim=-1).cpu().tolist()
                    label = batch['labels'].tolist()
                    preds.extend(pred)
                    gts.extend(label)
            val_acc = accuracy_score(gts, preds)
            print(
                f"Fold {fold + 1} | Epoch {ep + 1}/{epochs} | Train Loss={total_loss / len(train_dl):.4f} | Val Acc={val_acc:.4f}")

        fold_accuracies.append(val_acc)

    if fold_accuracies:
        print(f"\n📊 {k}-Fold 교차검증 평균 정확도: {sum(fold_accuracies) / len(fold_accuracies):.4f}")
    else:
        print("\n학습을 완료한 Fold가 없습니다.")


# =========================================================
# --- 메인 실행 ---
# =========================================================
if __name__ == "__main__":
    data_folder_for_training = "data"
    train_with_kfold(data_folder_for_training, epochs=10, k=5)

