import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_folder = 'data'
if not os.path.isdir(data_folder):
    print(f"❌ '{data_folder}' 폴더를 찾을 수 없습니다.")

files_to_process = glob.glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)

if not files_to_process:
    print("⚠️ 처리할 CSV 파일(.csv)을 찾을 수 없습니다.")

print(f"총 {len(files_to_process)}개의 CSV 파일을 처리합니다.\n")

all_true = []
all_pred = []

for filepath in files_to_process:
    try:
        print(f"📄 '{filepath}' 파일 작업 시작...")

        # 인코딩 처리
        try:
            df = pd.read_csv(filepath, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            print("  -> utf-8 디코딩 실패. euc-kr로 다시 시도합니다.")
            df = pd.read_csv(filepath, header=None, encoding='euc-kr')

        # # 컬럼 개수 확인 (최소 5개 필요)
        # if df.shape[1] < 5:
        #     print(f"  -> ⚠️ 스킵: '{filepath}' 은(는) 최소 5개 컬럼이 없음")
        #     continue

        # 숫자형 변환 (문자, NaN 제거)
        y_true = pd.to_numeric(df[4], errors='coerce')
        y_pred = pd.to_numeric(df[3], errors='coerce')

        # NaN 제거
        mask = ~y_true.isna() & ~y_pred.isna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    except Exception as e:
        print(f"  -> ❌ 파일 처리 중 오류 발생: {e}")

# numpy array 변환
all_true = np.array(all_true, dtype=float)
all_pred = np.array(all_pred, dtype=float)

if len(all_true) == 0:
    print("❌ 유효한 데이터가 없습니다.")
else:
    # MAE, MSE, RMSE 계산
    mae = mean_absolute_error(all_true, all_pred)
    mse = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)

    print("\n📊 최종 결과 (모든 데이터 합산 기준)")
    print("✅ 평균 절댓값 오차 (MAE):", mae)
    print("✅ 평균 제곱 오차 (MSE):", mse)
    print("✅ 제곱근 평균 제곱 오차 (RMSE):", rmse)
