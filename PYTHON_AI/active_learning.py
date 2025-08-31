import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
import glob
# ✅ 상세 분석을 위해 scikit-learn 임포트 추가
from sklearn.metrics import classification_report, confusion_matrix
# ✅ 클래스 가중치 계산을 위해 추가
from sklearn.utils.class_weight import compute_class_weight

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. 한글 폰트 설정 ---
try:
    font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    try:
        plt.rc('font', family='AppleGothic')
    except:
        print("한글 폰트를 찾을 수 없습니다. 그래프의 제목과 축이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False


# --- 2. 모델 클래스 정의 ---
class TextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits


# --- 3. 실제 데이터 로드 함수 (수정됨) ---
def load_data_for_interactive_al(labeled_folder, unlabeled_folder):
    """
    라벨링된 데이터와 라벨 없는 데이터를 각각의 폴더에서 로드합니다.
    """
    # 3a. 라벨링된 데이터 로드 (data/)
    print(f"'{labeled_folder}'에서 라벨링된 데이터를 로드합니다.")
    labeled_files = glob.glob(os.path.join(labeled_folder, '*.csv'))
    if not labeled_files:
        print(f"오류: '{labeled_folder}' 폴더에 CSV 파일이 없습니다.")
        return None, None

    labeled_dfs = []
    for f in labeled_files:
        try:
            df = pd.read_csv(f, header=None, skiprows=1, encoding='utf-8')
            labeled_dfs.append(df)
        except UnicodeDecodeError:
            df = pd.read_csv(f, header=None, skiprows=1, encoding='cp949')
            labeled_dfs.append(df)
    labeled_df_full = pd.concat(labeled_dfs, ignore_index=True)
    labeled_df_full = labeled_df_full[[0, 3]].rename(columns={0: 'text', 3: 'label'})

    # 3b. 라벨 없는 데이터 로드 (activeLearning/) - 댓글 내용(1열)만 가져옴
    print(f"'{unlabeled_folder}'에서 라벨 없는 후보 데이터를 로드합니다.")
    unlabeled_files = glob.glob(os.path.join(unlabeled_folder, '*.csv'))
    if not unlabeled_files:
        print(f"경고: '{unlabeled_folder}' 폴더에 파일이 없어, 액티브 러닝 후보가 없습니다.")
        unlabeled_pool_df = pd.DataFrame(columns=['text'])
    else:
        unlabeled_dfs = []
        for f in unlabeled_files:
            try:
                # usecols=[0]을 사용하여 첫 번째 열만 읽어옴
                df = pd.read_csv(f, header=None, skiprows=1, usecols=[0], encoding='utf-8')
                unlabeled_dfs.append(df)
            except UnicodeDecodeError:
                df = pd.read_csv(f, header=None, skiprows=1, usecols=[0], encoding='cp949')
                unlabeled_dfs.append(df)
        unlabeled_pool_df = pd.concat(unlabeled_dfs, ignore_index=True)
        unlabeled_pool_df.columns = ['text']

    # 3c. 텍스트 라벨을 숫자 ID로 변환
    label_map = {"칭찬": 0, "비판": 1, "작품내용": 2, "논쟁": 3, "관련없음": 4}
    labeled_df_full['label_id'] = labeled_df_full['label'].map(label_map)
    labeled_df_full.dropna(subset=['text', 'label_id'], inplace=True)
    labeled_df_full['label_id'] = labeled_df_full['label_id'].astype(int)

    return labeled_df_full, unlabeled_pool_df, label_map


# --- 4. 메인 인터랙티브 함수 ---
def run_interactive_active_learning():
    print("=" * 60)
    print("### ✍️ 인터랙티브 액티브 러닝 도구 시작 ###")
    print("=" * 60)

    # --- 파라미터 설정 ---
    LABELED_DATA_FOLDER = 'data'
    UNLABELED_DATA_FOLDER = 'activeLearning'
    TEST_SET_RATIO = 0.2
    VALIDATION_SET_RATIO = 0.2
    QUERY_SIZE = 7
    MODEL_NAME = "klue/bert-base"

    # ✅ --- GPU 사용 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 현재 사용 장치: {device} ---")

    # --- 4a. 데이터 준비 및 분할 ---
    print("\n--- 1단계: 데이터 준비 및 분할 ---")
    labeled_df_full, unlabeled_pool_df, label_map = load_data_for_interactive_al(LABELED_DATA_FOLDER,
                                                                                 UNLABELED_DATA_FOLDER)
    if labeled_df_full is None: return

    known_labeled_texts = set(labeled_df_full['text'])

    original_unlabeled_count = len(unlabeled_pool_df)
    unlabeled_pool_df = unlabeled_pool_df[~unlabeled_pool_df['text'].isin(known_labeled_texts)].reset_index(drop=True)
    if original_unlabeled_count > len(unlabeled_pool_df):
        print(
            f"  - 중복 제거: activeLearning 폴더에서 이미 라벨링된 댓글 {original_unlabeled_count - len(unlabeled_pool_df)}개를 제외했습니다.")

    remaining_df, test_df = train_test_split(labeled_df_full, test_size=TEST_SET_RATIO, random_state=42,
                                             stratify=labeled_df_full['label_id'])
    labeled_df, validation_df = train_test_split(remaining_df, test_size=VALIDATION_SET_RATIO / (1 - TEST_SET_RATIO),
                                                 random_state=42, stratify=remaining_df['label_id'])

    print(f"  - 테스트 세트: {len(test_df)}개 (최종 평가용)")
    print(f"  - 검증 세트: {len(validation_df)}개 (중간 점검용)")
    print(f"  - 초기 학습 세트: {len(labeled_df)}개")
    print(f"  - 라벨 없는 학습 후보(Unlabeled Pool): {len(unlabeled_pool_df)}개")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    validation_accuracies = []

    initial_labeled_count = len(labeled_df)

    num_rounds = len(unlabeled_pool_df) // QUERY_SIZE

    # ✅ --- 핵심 수정: 모델을 루프 밖에서 한 번만 생성 ---
    print("\n--- 2단계: 초기 모델 생성 ---")
    model = TextClassifier(MODEL_NAME, num_classes=5).to(device)

    print(f"\n--- 3단계: 액티브 러닝 루프 시작 (총 {num_rounds}라운드 예상) ---")
    for i in range(num_rounds + 1):
        current_labels_count = len(labeled_df)
        print(f"\n🔄 라운드 {i + 1} (현재 학습 라벨 수: {current_labels_count})")

        # --- 모델 학습 (추가 학습) ---
        if i == 0:
            print("  - 초기 데이터로 첫 모델을 학습합니다... (시간이 걸릴 수 있습니다)")
        else:
            print("  - 새로 추가된 데이터로 모델을 추가 학습(Fine-tuning)합니다...")

        train_texts = labeled_df['text'].tolist()
        train_labels = torch.tensor(labeled_df['label_id'].tolist(), dtype=torch.long)

        # ✅ --- 핵심 수정: 클래스 가중치 계산 ---
        # 데이터가 적은 클래스에 더 높은 가중치를 부여
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels.numpy()),
            y=train_labels.numpy()
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"  - 클래스 가중치 적용: {np.round(class_weights.cpu().numpy(), 2)}")

        optim = torch.optim.Adam(model.parameters(), lr=5e-6)
        # ✅ 손실 함수에 가중치 적용
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model.train()

        epochs = 10 if len(labeled_df) < 500 else 5
        batch_size = 16

        for epoch in range(epochs):
            permutation = torch.randperm(len(train_texts))
            for j in range(0, len(train_texts), batch_size):
                indices = permutation[j:j + batch_size]
                batch_texts = [train_texts[k] for k in indices]
                batch_labels = train_labels[indices].to(device)

                if not batch_texts: continue

                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(
                    device)
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

                loss = criterion(logits, batch_labels);
                optim.zero_grad();
                loss.backward();
                optim.step()

        # --- 성능 측정 ---
        model.eval()
        val_texts = validation_df['text'].tolist()
        val_labels = torch.tensor(validation_df['label_id'].tolist(), dtype=torch.long).to(device)
        all_preds = []
        with torch.no_grad():
            for j in range(0, len(val_texts), 32):
                batch_texts = val_texts[j:j + 32]
                if not batch_texts: continue
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(
                    device)
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)

        all_preds_tensor = torch.cat(all_preds)
        accuracy = (all_preds_tensor == val_labels).float().mean().item()
        validation_accuracies.append(accuracy)
        print(f"  📈 [모의고사] 검증 세트 전체 정확도: {accuracy:.2%}")

        # --- 상세 분석 리포트 출력 ---
        print("\n  --- 상세 분석 리포트 ---")
        y_true = val_labels.cpu().numpy()
        y_pred = all_preds_tensor.cpu().numpy()
        id_to_label_map = {v: k for k, v in label_map.items()}
        target_names = [id_to_label_map.get(i, "unknown") for i in range(len(label_map))]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

        print("  --- 혼동 행렬 (행: 실제, 열: 예측) ---")
        unique_labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
        cm_target_names = [id_to_label_map.get(i, "unknown") for i in unique_labels]
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        print(pd.DataFrame(cm, index=cm_target_names, columns=cm_target_names))
        print("-" * 25)

        if len(unlabeled_pool_df) < QUERY_SIZE: break

        # --- 조기 종료 기능 ---
        continue_labeling = input("\n라벨링을 계속하시겠습니까? (y/n): ").strip().lower()
        if continue_labeling != 'y':
            print("사용자 요청으로 라벨링을 중단합니다.")
            break

        # --- 헷갈리는 데이터 선별 ---
        print("  - 모델이 가장 헷갈리는 데이터를 선별합니다... (데이터가 많으면 시간이 걸립니다)")
        unlabeled_texts = unlabeled_pool_df['text'].dropna().tolist()
        all_uncertainties = []
        inf_batch_size = 32

        with torch.no_grad():
            for j in range(0, len(unlabeled_texts), inf_batch_size):
                batch_texts = unlabeled_texts[j:j + inf_batch_size]
                if not batch_texts: continue

                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(
                    device)
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

                probabilities = F.softmax(logits, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                uncertainty = 1 - max_probs
                all_uncertainties.append(uncertainty.cpu())

        if not all_uncertainties:
            print("  - 불확실성을 계산할 데이터가 없어 중단합니다.")
            break

        full_uncertainty_tensor = torch.cat(all_uncertainties)
        query_indices = torch.topk(full_uncertainty_tensor, k=min(QUERY_SIZE, len(unlabeled_texts))).indices

        questions_to_label = unlabeled_pool_df.iloc[query_indices.numpy()]

        # --- ⭐ 실제 사용자 라벨링 과정 ⭐ ---
        print("\n" + "-" * 20);
        print("✍️ 모델이 선택한 데이터에 라벨을 입력해주세요.");
        print("-" * 20)
        newly_labeled_rows = []
        valid_labels_text = list(label_map.keys())
        for index, row in questions_to_label.iterrows():
            while True:
                print(f"\n댓글: {row['text']}")
                user_input = input(f"라벨 입력 [{'/'.join(valid_labels_text)}]: ").strip()
                if user_input in valid_labels_text:
                    new_row = {'text': row['text'], 'label': user_input, 'label_id': label_map[user_input]}
                    newly_labeled_rows.append(new_row)
                    break
                else:
                    print(">> 잘못된 라벨입니다. 다시 입력해주세요. <<")

        # --- 데이터셋 업데이트 ---
        newly_labeled_df = pd.DataFrame(newly_labeled_rows)
        labeled_df = pd.concat([labeled_df, newly_labeled_df], ignore_index=True)
        unlabeled_pool_df = unlabeled_pool_df.drop(questions_to_label.index)
        print(f"\n  ✅ 라벨링 완료! 학습 세트에 {len(newly_labeled_df)}개의 데이터를 추가합니다.")

    # --- 최종 평가 ---
    print("\n\n--- 4단계: 최종 성능 평가 ---")
    test_texts = test_df['text'].tolist()
    test_labels = torch.tensor(test_df['label_id'].tolist(), dtype=torch.long).to(device)
    with torch.no_grad():
        inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        final_accuracy = (torch.argmax(logits, dim=1) == test_labels).float().mean().item()
    print(f"  🎓 [수능 시험] 최종 모델의 테스트 세트 최종 정확도: {final_accuracy:.2%}")

    # --- 결과 시각화 ---
    print("\n--- 5단계: 결과 시각화 ---")
    labeled_counts = [initial_labeled_count + i * QUERY_SIZE for i in range(len(validation_accuracies))]
    plt.figure(figsize=(10, 6));
    plt.plot(labeled_counts, validation_accuracies, marker='o', linestyle='-')
    plt.title('액티브 러닝 라벨 수에 따른 검증 세트 정확도 변화');
    plt.xlabel('학습에 사용된 라벨 데이터 개수')
    plt.ylabel('검증 세트 정확도');
    plt.grid(True);
    plt.xticks(labeled_counts, rotation=45)
    plt.ylim(0, max(1.0, max(validation_accuracies) * 1.2))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'));
    plt.tight_layout();
    plt.show()


# --- 5. 스크립트 실행 ---
if __name__ == "__main__":
    run_interactive_active_learning()

