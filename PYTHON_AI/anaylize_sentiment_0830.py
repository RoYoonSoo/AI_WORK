import warnings
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# pandas에서 openpyxl 관련 경고가 나올 수 있어 무시 처리합니다.
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------
# 1. 텍스트 데이터셋 클래스 정의 (✅ 수정된 부분)
# ---------------------------------------------------------
class TextCommentDataset(Dataset):
    def __init__(self, data_folder_path, tokenizer):
        self.tokenizer = tokenizer
        file_paths = glob.glob(os.path.join(data_folder_path, '**', '*.*'), recursive=True)
        file_paths = [f for f in file_paths if os.path.isfile(f) and not os.path.basename(f).startswith('.')]

        all_dfs = []
        for p in file_paths:
            try:
                # skiprows=1 추가: 첫 번째 행(헤더)을 건너뛰고 읽음
                if p.endswith('.csv'):
                    all_dfs.append(pd.read_csv(p, header=None, skiprows=1))
                elif p.endswith('.xlsx'):
                    all_dfs.append(pd.read_excel(p, header=None, skiprows=1))
            except Exception as e:
                print(f"파일을 읽는 중 오류 발생: {p}, 오류: {e}")

        if not all_dfs:
            # 처리할 데이터가 없는 경우 빈 DataFrame 생성
            combined_df = pd.DataFrame()
        else:
            combined_df = pd.concat(all_dfs, ignore_index=True)

        # 데이터가 없을 경우를 대비한 예외 처리
        if combined_df.empty:
            self.comments = []
            self.labels = []
            return

        self.comments = combined_df[0].fillna('').tolist()

        # 학습할 라벨을 E열(인덱스 4)에서 가져오도록 수정
        if 4 in combined_df.columns:
            self.label_map = {
                "칭찬": 0, "비판": 1, "작품내용": 2, "논쟁": 3, "관련없음": 4
            }
            self.labels = combined_df[4].map(self.label_map).fillna(-1).astype(int).tolist()
        else:
            # 오류 메시지도 E열 기준으로 수정
            raise ValueError("E열에 라벨 데이터가 없습니다. 1단계(라벨링)를 먼저 실행했는지 확인하세요.")

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        if self.labels[idx] == -1:
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        comment = self.comments[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            comment, return_tensors='pt', truncation=True, max_length=128, padding='max_length'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ---------------------------------------------------------
# 2. 텍스트 분류 모델 클래스 정의 (변경 없음)
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
# 3. 모델 학습 함수 정의 (변경 없음)
# ---------------------------------------------------------
def train_text_classifier(data_folder_path, epochs=5):
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TextClassifier(MODEL_NAME, num_classes=5)
    try:
        ds = TextCommentDataset(data_folder_path, tokenizer)
    except ValueError as e:
        print(f"⚠️ 데이터셋 생성 오류: {e}")
        return None
    if len(ds) == 0:
        print("⚠️ 학습할 데이터를 찾을 수 없습니다. `data` 폴더의 파일에 헤더 외 데이터가 있는지, 1단계 라벨링이 올바르게 실행되었는지 확인해주세요.")
        return None
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    print(f"\n🔥 총 {len(ds)}개의 댓글 데이터로 텍스트 모델 학습을 시작합니다...")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        processed_samples = 0
        for batch in dl:
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(logits, batch['labels'])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch['input_ids'].size(0)
            processed_samples += batch['input_ids'].size(0)

        if processed_samples > 0:
            print(f"Epoch {ep + 1}/{epochs}  loss={total_loss / processed_samples:.4f}")
        else:
            print(f"Epoch {ep + 1}/{epochs}  - 학습할 데이터가 없습니다.")

    print("✅ 텍스트 모델 학습 완료!")
    return model


# =========================================================
# --- 4. 데이터 라벨링 함수 정의 (✅ 수정된 부분) ---
# =========================================================
def label_all_csv_files():
    print("🧠 제로샷 NLI 모델을 로딩 중입니다... (GPU 사용 가능 시 시간이 다소 걸릴 수 있습니다)")
    try:
        model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        print(f"✅ 모델 로딩 완료! 현재 사용 장치: {device}")
    except Exception as e:
        print(f"❌ 모델 로딩 중 오류 발생: {e}")
        return

    data_folder = 'data'
    if not os.path.isdir(data_folder):
        print(f"❌ '{data_folder}' 폴더를 찾을 수 없습니다.")
        return

    files_to_process = glob.glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)
    if not files_to_process:
        print("⚠️ 처리할 CSV 파일(.csv)을 찾지 못했습니다.")
        return

    print(f"\n총 {len(files_to_process)}개의 CSV 파일을 제로샷 NLI 모델로 분석하여 라벨링합니다.")

    hypotheses = {
        "칭찬": "이 문장은 작품에 대한 칭찬이다.",
        "비판": "이 문장은 작품에 대한 비판이다.",
        "작품내용": "이 문장은 작품과 관련된 내용이다.",
        "논쟁": "이 문장은 댓글여론과 논쟁을 펼치고 있다.",
        "관련없음": "이 문장은 작품에 대해서 이야기하는 것이 아니다",
    }

    try:
        entailment_index = model.config.label2id['entailment']
    except KeyError:
        print("⚠️ 'entailment' 라벨을 모델 설정에서 찾을 수 없습니다. 기본값인 0을 사용합니다.")
        entailment_index = 0

    for filepath in files_to_process:
        try:
            print(f"\n📄 '{filepath}' 파일 작업을 시작합니다...")
            # header=None으로 먼저 전체 파일을 읽음 (헤더 포함)
            try:
                df = pd.read_csv(filepath, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                print("  -> utf-8 디코딩 실패. cp949로 다시 시도합니다.")
                df = pd.read_csv(filepath, header=None, encoding='cp949')

            if len(df) < 2:
                print("  -> 헤더만 있거나 데이터가 없어 건너뜁니다.")
                continue

            # 2행부터의 댓글(인덱스 1부터)을 분석 대상으로 삼음
            comments_to_process = df.iloc[1:, 0].fillna('').tolist()

            print(f"  -> {len(comments_to_process)}개의 댓글을 분석합니다...")
            predicted_labels = []

            for comment in comments_to_process:
                if not comment.strip():
                    predicted_labels.append("")
                    continue

                entailment_scores = {}
                for label_name, hypothesis in hypotheses.items():
                    inputs = tokenizer(comment, hypothesis, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        entailment_scores[label_name] = logits[0][entailment_index].item()

                best_label = max(entailment_scores, key=entailment_scores.get)
                predicted_labels.append(best_label)

            # 예측된 라벨을 E열(인덱스 4)의 2행부터 채워넣음
            # E열이 없을 경우를 대비하여 길이 조절
            if 4 not in df.columns:
                df[4] = pd.NA
            df.loc[1:, 4] = predicted_labels

            print(f"  -> 분석 결과를 원본 파일의 'E열'에 저장합니다...")
            df.to_csv(filepath, index=False, header=False, encoding='utf-8-sig')
            print(f"  -> ✅ 완료!")

        except Exception as e:
            print(f"  -> ❌ 파일 처리 중 오류 발생: {e}")


# =========================================================
# ---                 메인 코드 실행 부분                 ---
# =========================================================
if __name__ == "__main__":
    # 1단계 실행 후, 아래 2단계 코드의 주석을 해제하여 학습을 진행하세요.

    print("========================================")
    print("### 1단계: 데이터 라벨링 시작 ###")
    print("========================================")
    label_all_csv_files()
    print("\n\n")
    #
    # print("========================================")
    # print("### 2단계: 모델 학습 시작 ###")
    # print("========================================")
    # data_folder_for_training = "data"
    # my_trained_model = train_text_classifier(data_folder_for_training)
    #
    # if my_trained_model:
    #     print("\n🎉 나만의 댓글 분석 모델 학습에 성공했습니다!")