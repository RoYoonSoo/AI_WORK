import pandas as pd
from transformers import pipeline
import os
import glob

# --- ⚙️ 설정 부분 ---

# 'comments_'로 시작하고 '.csv'로 끝나는 모든 파일을 찾아서 정렬합니다.
# 이렇게 하면 comments_1.csv, comments_2.csv, ... 순서대로 처리됩니다.
FILE_PATHS = sorted(glob.glob("data/comments_*.csv"))
RESULT_DIR = "results"
RESULT_FILE = os.path.join(RESULT_DIR, "full_summary.txt")

# 1️⃣ 모델 불러오기
print("🔄 요약 모델을 불러오는 중입니다...")
# KoBART: 한줄 요약용
summarizer = pipeline("summarization", model="gogamza/kobart-summarization")
print("✅ 모델 불러오기 완료!")


# 2️⃣ 회차별 요약을 위한 텍스트 생성 함수 (sentiment 부분 제거)
def generate_summary_input_text(episode_df):
    """
    요약 모델에 넣기 좋은 형태로 회차별 텍스트를 가공합니다.
    """
    # 좋아요 많이 받은 상위 3개 댓글 선택 (댓글이 3개 미만일 경우도 처리)
    top_n = min(3, len(episode_df))
    if top_n == 0:
        return "이 회차에는 댓글이 없습니다."

    top_comments = episode_df.nlargest(top_n, 'likes')['comment'].tolist()
    top_comments_text = ". ".join(top_comments)

    # 요약 모델에 입력할 최종 텍스트 생성
    full_text = f"이 회차의 주요 반응은 다음과 같습니다: {top_comments_text}."
    return full_text


# 3️⃣ 회차별 요약 및 전체 요약 생성 (핵심 로직 수정)
episode_summaries = []

# 파일 목록이 비어있지 않은지 확인
if FILE_PATHS:
    print("\n--- 🚀 회차별 요약 시작 ---")
    # 파일을 하나씩 순회하며 요약 진행
    for i, path in enumerate(FILE_PATHS):
        episode_num = i + 1
        print(f"\n⏳ {episode_num}회차 ({os.path.basename(path)}) 댓글 요약 중...")

        try:
            # 현재 회차(파일)의 댓글만 읽기
            # 1, 3, 4번째 열만 읽고, 바로 새 이름 부여
            episode_df = pd.read_csv(
                path,
                encoding='utf-8-sig',
                header=None,
                skiprows=1,
                usecols=[0, 2, 3],
                names=['comment', 'likes', 'dislikes']
            )

            # 파일은 있지만 내용이 비어있는 경우 건너뛰기
            if episode_df.empty:
                print(f"⚠️ {episode_num}회차({os.path.basename(path)}) 파일에 내용이 없어 건너뜁니다.")
                continue

            # 1. 요약할 텍스트 생성
            input_text = generate_summary_input_text(episode_df)

            # 2. 모델을 통해 요약 (max_length -> max_new_tokens 로 변경)
            summary = summarizer(input_text, max_new_tokens=60, min_new_tokens=10, do_sample=False)[0]["summary_text"]
            episode_summaries.append(summary)

            # 3. 회차별 요약 결과 출력
            print(f"✅ {episode_num}회차 한 줄 요약: {summary}")

        except Exception as e:
            print(f"❌ '{path}' 파일 처리 중 오류 발생: {e}")

    # 4️⃣ 전체 요약 및 파일 저장
    if episode_summaries:
        print("\n--- 🚀 전체 요약 시작 ---")
        # 1. 회차별 요약들을 합쳐서 전체 요약의 입력으로 사용
        overall_input_text = " ".join(episode_summaries)

        # 2. 전체 내용 요약 (max_length -> max_new_tokens 로 변경)
        print("⏳ 모든 회차의 내용을 종합하여 요약 중...")
        overall_summary = summarizer(overall_input_text, max_new_tokens=100, min_new_tokens=20, do_sample=False)[0][
            "summary_text"]

        # 3. 최종 결과 출력
        print("\n🎉 최종 요약 결과입니다! 🎉")
        print("---")
        for i, summary in enumerate(episode_summaries):
            print(f"📄 {i + 1}회차: {summary}")
        print("---")
        print(f"✨ 전체: {overall_summary}")

        # 5. 파일로 저장
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            f.write("--- 회차별 한 줄 요약 ---\n")
            for i, summary in enumerate(episode_summaries):
                f.write(f"{i + 1}회차: {summary}\n")
            f.write("\n--- 전체 종합 요약 ---\n")
            f.write(overall_summary)

        print(f"\n✅ 전체 요약 결과가 '{RESULT_FILE}' 파일에 저장되었습니다.")
    else:
        print("⚠️ 요약된 내용이 없어 전체 요약을 건너뜁니다.")
else:
    print("⚠️ 처리할 'comments_*.csv' 파일을 찾지 못했습니다. 파일 이름과 위치를 확인해주세요.")

