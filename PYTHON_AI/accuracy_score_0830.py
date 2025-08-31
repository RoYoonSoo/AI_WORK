import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# --- 5. 예측 평가 함수 정의 (✅ 새로 추가된 부분) ---
# =========================================================
def evaluate_predictions():
    print("📊 D열(정답)과 E열(예측)을 비교하여 모델 성능을 평가합니다...")

    data_folder = 'data'
    file_paths = glob.glob(os.path.join(data_folder, '**', '*.*'), recursive=True)
    file_paths = [f for f in file_paths if os.path.isfile(f) and not os.path.basename(f).startswith('.')]

    if not file_paths:
        print("⚠️ 평가할 파일을 찾을 수 없습니다.")
        return

    all_dfs = []
    for p in file_paths:
        try:
            # skiprows=1: 첫 번째 행(헤더)은 건너뛰고 읽음
            if p.endswith('.csv'):
                all_dfs.append(pd.read_csv(p, header=None, skiprows=1))
            elif p.endswith('.xlsx'):
                all_dfs.append(pd.read_excel(p, header=None, skiprows=1))
        except Exception as e:
            print(f"파일을 읽는 중 오류 발생: {p}, 오류: {e}")

    if not all_dfs:
        print("⚠️ 파일에서 데이터를 읽어오지 못했습니다.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # D열(3) 또는 E열(4)이 없거나 비어있는 경우 평가 중단
    if 3 not in combined_df.columns or 4 not in combined_df.columns:
        print("❌ 평가에 필요한 D열 또는 E열이 모든 파일에 존재하지 않습니다.")
        return

    # D열 또는 E열에 데이터가 없는 행은 평가에서 제외
    combined_df.dropna(subset=[3, 4], inplace=True)

    if combined_df.empty:
        print("⚠️ 비교할 데이터가 없습니다. D열과 E열에 내용이 있는지 확인해주세요.")
        return

    # 데이터 타입을 문자열로 통일하고 공백 제거
    y_true = combined_df[3].astype(str).str.strip()
    y_pred = combined_df[4].astype(str).str.strip()

    # --- 1. 정확도 (Accuracy) ---
    # 사용자가 요청한 'D열과 E열이 같은 비율'
    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"### 1. 전체 정확도 (Accuracy) ###")
    print(f"D열과 E열의 라벨이 일치하는 비율은 약 {accuracy:.2%} 입니다.")
    print(f"(총 {len(y_true)}개의 데이터 중 {int(accuracy * len(y_true))}개 일치)")
    print("=" * 50)

    # --- 2. 분류 리포트 (Classification Report) ---
    # 각 라벨별 정밀도, 재현율, F1 점수를 보여줌
    print("\n" + "=" * 50)
    print("### 2. 상세 분류 리포트 ###")
    print("정밀도(Precision): 모델이 'A'라고 예측한 것 중 실제 'A'인 비율")
    print("재현율(Recall)   : 실제 'A'인 것들 중 모델이 'A'라고 맞춘 비율")
    print("F1-Score         : 정밀도와 재현율의 조화 평균 (높을수록 좋음)")
    print("-" * 50)

    # 라벨 순서를 정하기 위해 고유 라벨 목록 추출
    labels = sorted(pd.concat([y_true, y_pred]).unique())
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    print("=" * 50)

    # --- 3. 혼동 행렬 (Confusion Matrix) ---
    # 모델이 어떤 라벨을 어떤 라벨로 혼동하는지 보여줌
    print("\n" + "=" * 50)
    print("### 3. 혼동 행렬 (Confusion Matrix) ###")
    print("행(세로): 실제 정답 라벨, 열(가로): 모델의 예측 라벨")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"실제: {l}" for l in labels], columns=[f"예측: {l}" for l in labels])
    print(cm_df)
    print("\n(예: '실제: 칭찬' 행의 '예측: 비판' 열 값이 10이라면, 실제 '칭찬'인 댓글 10개를 '비판'으로 잘못 예측했다는 의미입니다.)")
    print("=" * 50)

# =========================================================
# ---                 메인 코드 실행 부분                 ---
# =========================================================
if __name__ == "__main__":
    # 1단계: 라벨링 결과 평가 (D열 vs E열)
    print("========================================")
    print("### 3단계: 예측 결과 평가 시작 ###")
    print("========================================")
    evaluate_predictions()
    print("\n\n")