import pandas as pd
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, pipeline
import warnings
import openpyxl

# pandas에서 openpyxl 관련 경고가 나올 수 있어 무시 처리합니다.
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------
# 1. 텍스트 데이터셋 클래스 정의
# ---------------------------------------------------------
class TextCommentDataset(Dataset):
    def __init__(self, data_folder_path, tokenizer):
        self.tokenizer = tokenizer

        # --- ⚠️ 중요 수정사항 ---
        # data_folder_path와 그 모든 하위 폴더를 순회하며 파일을 찾도록 수정
        file_paths = glob.glob(os.path.join(data_folder_path, '**', '*.*'), recursive=True)
        file_paths = [f for f in file_paths if os.path.isfile(f) and not os.path.basename(f).startswith('.')]

        all_dfs = []
        for p in file_paths:
            try:
                if p.endswith('.csv'):
                    all_dfs.append(pd.read_csv(p, header=None))
                elif p.endswith('.xlsx'):
                    all_dfs.append(pd.read_excel(p, header=None))
            except Exception as e:
                print(f"파일을 읽는 중 오류 발생: {p}, 오류: {e}")

        combined_df = pd.concat(all_dfs, ignore_index=True)

        self.comments = combined_df[0].fillna('').tolist()
        # D열(인덱스 3)이 없는 경우를 대비하여 오류 처리
        if 3 in combined_df.columns:
            self.labels = (combined_df[3].values + 5).astype(int)
        else:
            raise ValueError("D열에 점수 데이터가 없습니다. 1단계(라벨링)를 먼저 실행했는지 확인하세요.")

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
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
# 2. 텍스트 분류 모델 클래스 정의
# ---------------------------------------------------------
class TextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=11):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits


# ---------------------------------------------------------
# 3. 모델 학습 함수 정의
# ---------------------------------------------------------
def train_text_classifier(data_folder_path, epochs=5):
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TextClassifier(MODEL_NAME, num_classes=11)

    try:
        ds = TextCommentDataset(data_folder_path, tokenizer)
    except ValueError as e:
        print(f"⚠️ 데이터셋 생성 오류: {e}")
        return None

    if len(ds) == 0:
        print("⚠️ 학습할 데이터를 찾을 수 없습니다. `data` 폴더에 라벨링된 파일이 있는지 확인해주세요.")
        return None

    dl = DataLoader(ds, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)

    print(f"\n🔥 총 {len(ds)}개의 댓글 데이터로 텍스트 모델 학습을 시작합니다...")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dl:
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(logits, batch['labels'])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch['input_ids'].size(0)
        print(f"Epoch {ep + 1}/{epochs}  loss={total_loss / len(ds):.4f}")

    print("✅ 텍스트 모델 학습 완료!")
    return model


# ---------------------------------------------------------
# 4. 데이터 라벨링 함수 정의
# ---------------------------------------------------------
def label_all_csv_files():
    print("🧠 두 개의 전문 분석 모델을 로딩 중입니다...")
    try:
        relevance_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        sentiment_analyzer = pipeline('sentiment-analysis', model='sangrimlee/bert-base-multilingual-cased-nsmc')
        print("✅ 모든 모델 로딩이 완료되었습니다.")
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

    print(f"\n총 {len(files_to_process)}개의 CSV 파일을 2단계로 정밀 분석합니다.")
    for filepath in files_to_process:
        try:
            print(f"\n📄 '{filepath}' 파일 작업을 시작합니다...")

            # --- ⚠️ 중요 수정사항: 인코딩 처리 최종 강화 ---
            try:
                # 1순위: 가장 표준 방식인 utf-8로 읽기를 시도합니다.
                df = pd.read_csv(filepath, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                # 2순위: utf-8 실패 시, 윈도우 환경을 고려해 cp949로 읽기를 시도합니다.
                print("  -> utf-8 디코딩 실패. cp949로 다시 시도합니다.")
                df = pd.read_csv(filepath, header=None, encoding='cp949')

            comments = df[0].fillna('').tolist()
            if not comments:
                print("  -> 댓글 내용이 없어 건너뜁니다.")
                continue

            print(f"  -> {len(comments)}개의 댓글을 분석합니다...")

            final_scores = []
            candidate_labels = ["웹툰 평가", "기타 내용"]

            for comment in comments:
                if not comment.strip():
                    final_scores.append(-1.0)
                    continue

                relevance_result = relevance_classifier(comment, candidate_labels)
                top_label = relevance_result['labels'][0]
                top_score = relevance_result['scores'][0]

                # if top_label == "기타 내용" and top_score > 0.7:
                #     final_scores.append(0.0)
                # else:
                sentiment_result = sentiment_analyzer(comment)[0]
                score = sentiment_result['score']

                if sentiment_result['label'].lower() == 'positive':
                    final_scores.append(score*5)
                else:
                    final_scores.append(-score*5)

            df[3] = final_scores

            print(f"  -> 분석 결과를 원본 파일에 저장합니다...")
            df.to_csv(filepath, index=False, header=False, encoding='utf-8-sig')
            print(f"  -> ✅ 완료!")

        except Exception as e:
            print(f"  -> ❌ 파일 처리 중 오류 발생: {e}")
# =========================================================
# ---                 메인 코드 실행 부분                 ---
# =========================================================
if __name__ == "__main__":

    # --- 1단계: 원본 데이터에 점수 기록하기 ---
    # 'data' 폴더와 그 하위 폴더의 모든 파일 D열에 댓글 점수를 기록합니다.
    # 이 단계를 먼저 실행해서 학습 데이터를 준비해야 합니다.
    # (이미 점수 기록을 완료했다면 이 부분을 주석 처리(#)하고 2단계만 실행해도 됩니다.)
    print("========================================")
    print("### 1단계: 데이터 라벨링 시작 ###")
    print("========================================")
    label_all_csv_files()
    print("\n\n")  # 단계 구분을 위한 줄바꿈

    # # --- 2단계: 라벨링된 데이터로 나만의 모델 학습하기 ---
    # # 1단계에서 점수가 기록된 'data' 폴더 전체를 학습 데이터로 사용합니다.
    # print("========================================")
    # print("### 2단계: 모델 학습 시작 ###")
    # print("========================================")
    # data_folder_for_training = "data"
    # my_trained_model = train_text_classifier(data_folder_for_training)
    #
    # # 이제 'my_trained_model' 변수에 나만의 학습된 모델이 담겨 있습니다.
    # # 이 모델을 저장하거나 다른 새로운 댓글의 점수를 예측하는 데 사용할 수 있습니다.
    # if my_trained_model:
    #     print("\n🎉 나만의 댓글 분석 모델 학습에 성공했습니다!")