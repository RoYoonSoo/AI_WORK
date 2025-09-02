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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import copy

# --- 경고 메시지 무시 및 한글 폰트 설정 ---
warnings.filterwarnings('ignore')
try:
    font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    try:
        plt.rc('font', family='AppleGothic')
    except:
        print("한글 폰트를 찾을 수 없습니다. 그래프의 제목과 축이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False


# --- Focal Loss 클래스 정의 (이전과 동일) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- 모델 클래스 정의 (이전과 동일) ---
class TextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_vector = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token_vector)
        return logits


# --- 데이터 로드 함수 (이전과 동일) ---
def load_data(human_labeled_folder, unlabeled_folder, label_map):
    print(f"'{human_labeled_folder}'에서 사람이 라벨링한 데이터를 로드합니다.")
    labeled_files = glob.glob(os.path.join(human_labeled_folder, '*.csv'))
    if not labeled_files:
        print(f"오류: '{human_labeled_folder}' 폴더에 CSV 파일이 없습니다.")
        return None, None, None
    labeled_dfs = []
    for f in labeled_files:
        try:
            df = pd.read_csv(f, header=None, skiprows=1, on_bad_lines='skip', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(f, header=None, skiprows=1, on_bad_lines='skip', encoding='cp949')
        labeled_dfs.append(df)
    human_labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    human_labeled_df = human_labeled_df.iloc[:, [0, 3]].rename(columns={0: 'text', 3: 'label'})
    human_labeled_df.dropna(subset=['text', 'label'], inplace=True)
    human_labeled_df['text'] = human_labeled_df['text'].astype(str)
    human_labeled_df = human_labeled_df[human_labeled_df['text'].str.strip() != '']
    human_labeled_df['label_id'] = human_labeled_df['label'].map(label_map)
    human_labeled_df.dropna(subset=['label_id'], inplace=True)
    human_labeled_df['label_id'] = human_labeled_df['label_id'].astype(int)
    print(f"'{unlabeled_folder}'에서 라벨 없는 후보 데이터를 로드합니다.")
    unlabeled_files = glob.glob(os.path.join(unlabeled_folder, '*.csv'))
    if not unlabeled_files:
        unlabeled_pool_df = pd.DataFrame(columns=['text'])
    else:
        unlabeled_dfs = []
        for f in unlabeled_files:
            try:
                df = pd.read_csv(f, header=None, skiprows=1, usecols=[0], on_bad_lines='skip', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(f, header=None, skiprows=1, usecols=[0], on_bad_lines='skip', encoding='cp949')
            unlabeled_dfs.append(df)
        unlabeled_pool_df = pd.concat(unlabeled_dfs, ignore_index=True)
        unlabeled_pool_df.columns = ['text']
        unlabeled_pool_df.dropna(subset=['text'], inplace=True)
        unlabeled_pool_df['text'] = unlabeled_pool_df['text'].astype(str)
        unlabeled_pool_df = unlabeled_pool_df[unlabeled_pool_df['text'].str.strip() != '']
    return human_labeled_df, unlabeled_pool_df, label_map


# --- 앙상블 예측 함수 (프롬프트 적용) ---
def ensemble_predict(model, snapshots, tokenizer, texts, device, prompt, batch_size=32):
    model.to(device)
    all_avg_probs = None

    with torch.no_grad():
        for i, state_dict in enumerate(snapshots):
            model.load_state_dict(state_dict)
            model.eval()

            all_probs_snapshot = []
            for j in range(0, len(texts), batch_size):
                batch_texts = texts[j:j + batch_size]
                if not batch_texts: continue

                # [프롬프트 적용]
                batch_texts_with_prompt = [prompt + text for text in batch_texts]
                inputs = tokenizer(batch_texts_with_prompt, return_tensors='pt', padding=True, truncation=True,
                                   max_length=128).to(device)

                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                probabilities = F.softmax(logits, dim=1)
                all_probs_snapshot.append(probabilities.cpu())

            all_probs_tensor = torch.cat(all_probs_snapshot)
            if all_avg_probs is None:
                all_avg_probs = all_probs_tensor
            else:
                all_avg_probs += all_probs_tensor

    final_avg_probs = all_avg_probs / len(snapshots)
    final_preds = torch.argmax(final_avg_probs, dim=1)
    return final_preds


# --- 메인 인터랙티브 함수 ---
def run_advanced_pipeline():
    print("=" * 60);
    print("### 🚀 최종 성능 개선 파이프라인 시작 (프롬프트 엔지니어링 버전) ###");
    print("=" * 60)

    # --- 하이퍼파라미터 ---
    HUMAN_LABELED_FOLDER, UNLABELED_DATA_FOLDER = 'data', 'activeLearning'
    TEST_SET_RATIO, VALIDATION_SET_RATIO = 0.2, 0.2
    MODEL_NAME = "klue/roberta-base"  # 모델 교체
    QUERY_SIZE = 10
    MINORITY_CLASS_IDS = [3, 4]
    PSEUDO_LABEL_THRESHOLD = 0.98
    MAX_PSEUDO_PER_CLASS = 70
    FOCAL_LOSS_GAMMA = 2.0
    ENSEMBLE_EPOCHS = 8

    # [추가] 프롬프트 정의
    PROMPT = "댓글을 분류하세요. 특히 웹툰 내용에 대한 댓글과, 작가나 그림에 대한 비판을 구분하는 데 집중하세요: "

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 현재 사용 장치: {device} | 모델: {MODEL_NAME} ---")

    # --- 데이터 준비 ---
    print("\n--- 1단계: 데이터 준비 및 분할 ---")
    label_map = {"칭찬": 0, "비판": 1, "작품내용": 2, "논쟁": 3, "관련없음": 4}
    human_labeled_df, unlabeled_pool_df, _ = load_data(HUMAN_LABELED_FOLDER, UNLABELED_DATA_FOLDER, label_map)
    if human_labeled_df is None: return
    remaining_df, test_df = train_test_split(human_labeled_df, test_size=TEST_SET_RATIO, random_state=42,
                                             stratify=human_labeled_df['label_id'])
    labeled_df, validation_df = train_test_split(remaining_df, test_size=VALIDATION_SET_RATIO / (1 - TEST_SET_RATIO),
                                                 random_state=42, stratify=remaining_df['label_id'])
    known_labeled_texts = set(human_labeled_df['text'])
    unlabeled_pool_df = unlabeled_pool_df[~unlabeled_pool_df['text'].isin(known_labeled_texts)].reset_index(drop=True)
    print(f"  - 테스트: {len(test_df)}개 | 검증: {len(validation_df)}개 | 초기 학습: {len(labeled_df)}개")
    print(f"  - 라벨 없는 후보(Unlabeled Pool): {len(unlabeled_pool_df)}개")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    validation_accuracies = []
    initial_labeled_count = len(labeled_df)
    num_rounds = len(unlabeled_pool_df) // QUERY_SIZE if QUERY_SIZE > 0 else 0

    # --- 모델 및 옵티마이저 생성 ---
    print(f"\n--- 2단계: 메인 분류 모델({MODEL_NAME}) 생성 ---")
    model = TextClassifier(MODEL_NAME, num_classes=len(label_map)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # --- 액티브 러닝 루프 ---
    print(f"\n--- 3단계: 액티브 러닝 루프 시작 (총 {num_rounds}라운드 예상) ---")
    for i in range(num_rounds + 1):
        print(f"\n🔄 라운드 {i + 1} (현재 학습 라벨 수: {len(labeled_df)})")

        # --- 학습 ---
        train_texts = labeled_df['text'].tolist()
        train_labels = torch.tensor(labeled_df['label_id'].tolist(), dtype=torch.long)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels.numpy()),
                                             y=train_labels.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=class_weights)

        model_snapshots = []
        print(f"  - 모델 학습 및 스냅샷 앙상블 생성 (총 {ENSEMBLE_EPOCHS} 에포크)...")
        for epoch in range(ENSEMBLE_EPOCHS):
            model.train()
            permutation = torch.randperm(len(train_texts))
            for j in range(0, len(train_texts), 8):  # 배치 사이즈 조절 (roberta-large)
                indices = permutation[j:j + 8]
                batch_texts = [train_texts[k] for k in indices]
                batch_labels = train_labels[indices].to(device)
                if not batch_texts: continue

                # [프롬프트 적용]
                batch_texts_with_prompt = [PROMPT + text for text in batch_texts]
                inputs = tokenizer(batch_texts_with_prompt, return_tensors='pt', padding=True, truncation=True,
                                   max_length=128).to(device)

                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                loss = criterion(logits, batch_labels)
                optim.zero_grad();
                loss.backward();
                optim.step()

            print(f"    - 에포크 {epoch + 1} 완료, 스냅샷 저장.")
            model_snapshots.append(copy.deepcopy(model.state_dict()))

        # --- 자율 학습 (프롬프트 적용) ---
        print("  - 🧠 자율 학습 시작...")
        model.eval()
        pseudo_labeled_rows = []
        unlabeled_texts_for_pseudo = unlabeled_pool_df['text'].dropna().tolist()
        if unlabeled_texts_for_pseudo:
            with torch.no_grad():
                for j in range(0, len(unlabeled_texts_for_pseudo), 16):  # 배치 사이즈 조절
                    batch_texts = unlabeled_texts_for_pseudo[j:j + 16]
                    if not batch_texts: continue

                    # [프롬프트 적용]
                    batch_texts_with_prompt = [PROMPT + text for text in batch_texts]
                    inputs = tokenizer(batch_texts_with_prompt, return_tensors='pt', padding=True, truncation=True,
                                       max_length=128).to(device)

                    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    probabilities = F.softmax(logits, dim=1)
                    max_probs, preds = torch.max(probabilities, dim=1)
                    confident_mask = max_probs > PSEUDO_LABEL_THRESHOLD
                    if confident_mask.any():
                        confident_texts = np.array(batch_texts)[confident_mask.cpu().numpy()]
                        confident_preds = preds[confident_mask].cpu().numpy()
                        for text, pred_id in zip(confident_texts, confident_preds):
                            pseudo_labeled_rows.append({'text': text, 'label_id': pred_id})

        if pseudo_labeled_rows:
            pseudo_df = pd.DataFrame(pseudo_labeled_rows)
            pseudo_df = pseudo_df.groupby('label_id').head(MAX_PSEUDO_PER_CLASS).reset_index(drop=True)
            print(f"    - 모델이 확신하는 데이터 {len(pseudo_df)}개를 의사 라벨로 생성하여 추가 학습합니다.")

            augmented_texts = labeled_df['text'].tolist() + pseudo_df['text'].tolist()
            augmented_labels = torch.cat([torch.tensor(labeled_df['label_id'].tolist(), dtype=torch.long),
                                          torch.tensor(pseudo_df['label_id'].tolist(), dtype=torch.long)])
            model.train()
            for epoch in range(2):
                permutation = torch.randperm(len(augmented_texts))
                for j in range(0, len(augmented_texts), 8):  # 배치 사이즈 조절
                    indices = permutation[j:j + 8]
                    batch_texts = [augmented_texts[k] for k in indices]
                    batch_labels = augmented_labels[indices].to(device)
                    if not batch_texts: continue

                    # [프롬프트 적용]
                    batch_texts_with_prompt = [PROMPT + text for text in batch_texts]
                    inputs = tokenizer(batch_texts_with_prompt, return_tensors='pt', padding=True, truncation=True,
                                       max_length=128).to(device)

                    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    loss = criterion(logits, batch_labels)
                    optim.zero_grad();
                    loss.backward();
                    optim.step()
        else:
            print("    - 신뢰도 높은 의사 라벨을 찾지 못해 자율 학습을 건너뜁니다.")

        # --- 성능 측정 (앙상블 예측) ---
        val_texts, val_labels_cpu = validation_df['text'].tolist(), validation_df['label_id'].tolist()

        print("  - 앙상블 모델로 검증 세트 성능 측정 중...")
        all_preds_tensor = ensemble_predict(model, model_snapshots, tokenizer, val_texts, device, PROMPT)

        accuracy = accuracy_score(val_labels_cpu, all_preds_tensor.numpy())
        validation_accuracies.append(accuracy)
        print(f"  📈 [모의고사] 검증 세트 전체 정확도 (앙상블): {accuracy:.2%}")

        target_names = [name for name, id in sorted(label_map.items(), key=lambda item: item[1])]
        y_true = np.array(val_labels_cpu)
        y_pred = all_preds_tensor.cpu().numpy()
        print("\n  --- 상세 분석 리포트 ---")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        print("  --- 혼동 행렬 (행: 실제, 열: 예측) ---")
        cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map.values()))
        print(pd.DataFrame(cm, index=target_names, columns=target_names))
        print("-" * 25)

        if len(unlabeled_pool_df) < QUERY_SIZE:
            print("\n모든 데이터를 라벨링하여 루프를 종료합니다.")
            break
        if input("\n라벨링을 계속하시겠습니까? (y/n): ").strip().lower() != 'y':
            break

        # --- 데이터 선별 (프롬프트 적용) ---
        print("  - 🧠 데이터 선별 시작 (Uncertainty + Minority Targeting)...")
        model.eval()
        unlabeled_texts = unlabeled_pool_df['text'].dropna().tolist()
        all_probs = []
        with torch.no_grad():
            for j in range(0, len(unlabeled_texts), 16):  # 배치 사이즈 조절
                batch_texts = unlabeled_texts[j:j + 16]
                if not batch_texts: continue

                # [프롬프트 적용]
                batch_texts_with_prompt = [PROMPT + text for text in batch_texts]
                inputs = tokenizer(batch_texts_with_prompt, return_tensors='pt', padding=True, truncation=True,
                                   max_length=128).to(device)

                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                all_probs.append(F.softmax(logits, dim=1).cpu())
        all_probs_tensor = torch.cat(all_probs)
        # ... (이하 데이터 선별 및 사용자 라벨링 로직은 이전과 동일)
        entropy = -torch.sum(all_probs_tensor * torch.log(all_probs_tensor + 1e-9), dim=1)
        uncertain_indices = torch.topk(entropy, k=min(QUERY_SIZE, len(unlabeled_texts))).indices
        max_probs, preds = torch.max(all_probs_tensor, dim=1)
        minority_mask = torch.isin(preds, torch.tensor(MINORITY_CLASS_IDS))
        candidate_indices = torch.where(minority_mask)[0]
        sorted_minority_indices = candidate_indices[torch.argsort(max_probs[candidate_indices], descending=True)]
        final_query_indices = list(uncertain_indices.numpy()[:QUERY_SIZE // 2])
        for idx in sorted_minority_indices.numpy():
            if len(final_query_indices) >= QUERY_SIZE: break
            if idx not in final_query_indices: final_query_indices.append(idx)
        i = 0
        while len(final_query_indices) < QUERY_SIZE and i < len(uncertain_indices):
            idx = uncertain_indices[i].item()
            if idx not in final_query_indices: final_query_indices.append(idx)
            i += 1
        questions_to_label = unlabeled_pool_df.iloc[final_query_indices]

        # --- 사용자 라벨링 ---
        print("\n" + "-" * 20);
        print("✍️ 모델이 선택한 데이터에 라벨을 입력해주세요.");
        print("-" * 20)
        newly_labeled_rows = []
        for _, row in questions_to_label.iterrows():
            while True:
                print(f"\n댓글: {row['text']}")
                user_input = input(f"라벨 입력 [{'/'.join(label_map.keys())}]: ").strip()
                if user_input in label_map:
                    new_row = row.to_dict();
                    new_row['label_id'] = label_map[user_input]
                    newly_labeled_rows.append(new_row)
                    break
                else:
                    print(">> 잘못된 라벨입니다. 다시 입력해주세요. <<")
        labeled_df = pd.concat([labeled_df, pd.DataFrame(newly_labeled_rows)], ignore_index=True)
        unlabeled_pool_df = unlabeled_pool_df.drop(questions_to_label.index)

    # --- 최종 평가 (앙상블 예측) ---
    print("\n\n--- 4단계: 최종 성능 평가 ---")
    test_texts = test_df['text'].tolist()
    test_labels_cpu = test_df['label_id'].tolist()

    print("  - 최종 앙상블 모델로 테스트 세트 성능 측정 중...")
    final_preds = ensemble_predict(model, model_snapshots, tokenizer, test_texts, device, PROMPT).numpy()

    final_accuracy = accuracy_score(test_labels_cpu, final_preds)
    print(f"  🎓 [수능 시험] 최종 모델의 테스트 세트 최종 정확도 (앙상블): {final_accuracy:.2%}")

    # --- 최종 분석 및 시각화 ---
    print("\n" + "=" * 60);
    print("### 🔍 5단계: 최종 모델 오류 분석 (오답노트) ###");
    print("=" * 60)
    id_to_label_map = {v: k for k, v in label_map.items()}
    target_names = [name for name, id in sorted(label_map.items(), key=lambda item: item[1])]
    print(classification_report(test_labels_cpu, final_preds, target_names=target_names, zero_division=0))
    misclassified_indices = np.where(np.array(test_labels_cpu) != final_preds)[0]
    if len(misclassified_indices) > 0:
        print("\n--- 🔍 잘못 예측된 댓글 예시 (최대 10개) ---")
        for idx in misclassified_indices[:10]:
            print(
                f"\n  - 실제: {id_to_label_map.get(test_labels_cpu[idx], 'N/A')}, 예측: {id_to_label_map.get(final_preds[idx], 'N/A')}")
            print(f"  - 댓글: {test_df.iloc[idx]['text']}")
        print("-" * 20)
    print("\n--- 6단계: 결과 시각화 ---")
    labeled_counts = [initial_labeled_count + i * QUERY_SIZE for i in range(len(validation_accuracies))]
    plt.figure(figsize=(10, 6));
    plt.plot(labeled_counts, validation_accuracies, marker='o', linestyle='-');
    plt.title('액티브 러닝 라벨 수에 따른 검증 세트 정확도 변화');
    plt.xlabel('학습에 사용된 라벨 데이터 개수');
    plt.ylabel('검증 세트 정확도');
    plt.grid(True);
    plt.xticks(labeled_counts, rotation=45);
    plt.ylim(0, max(1.0, max(validation_accuracies, default=0) * 1.2));
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'));
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    run_advanced_pipeline()