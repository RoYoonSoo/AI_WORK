import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ✅ 한 회차 댓글 크롤링 (Selenium 사용) - 사용자님의 원본 코드로 복원
def crawl_comments(webtoon_id, episode_no, week, driver):
    """
    특정 웹툰 회차의 댓글, 좋아요, 싫어요 수를 크롤링합니다.
    """
    url = f"https://comic.naver.com/webtoon/detail?titleId={webtoon_id}&no={episode_no}&week={week}"
    driver.get(url)

    try:
        # --- 모든 댓글이 로드될 때까지 스크롤 ---
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, '_text_mfm2s_16'))
        )
        time.sleep(1)

        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    except Exception as e:
        print(f"❌ {episode_no}화 댓글 로딩 실패: {e}")
        return []

    soup = BeautifulSoup(driver.page_source, "html.parser")
    # 사용자님의 원본 코드 방식: p 태그를 먼저 찾음
    comment_list_items = soup.find_all('p', class_='_text_mfm2s_16')

    comments_data = []
    for li in comment_list_items:
        try:
            comment = li.get_text(strip=True).replace("BEST", "")

            # p 태그의 부모인 li를 찾아 올라가는 방식
            parent_li = li.find_parent('li', class_='_root_1koau_1')
            if not parent_li:
                continue  # 부모를 찾지 못하면 건너뛰기

            reaction_divs = parent_li.find_all('div', class_='_inside_2v7c9_21')

            likes = "0"
            dislikes = "0"

            if len(reaction_divs) > 1:
                like_spans = reaction_divs[1].find_all('span')
                if len(like_spans) > 1:
                    likes = like_spans[1].get_text(strip=True)

            if len(reaction_divs) > 2:
                dislike_spans = reaction_divs[2].find_all('span')
                if len(dislike_spans) > 1:
                    dislikes = dislike_spans[1].get_text(strip=True)

            comments_data.append({
                "comment": comment,
                "likes": likes,
                "dislikes": dislikes
            })
        except Exception as e:
            # 개별 댓글 파싱 오류 시 건너뛰기
            continue

    return comments_data


# ✅ 여러 회차 크롤링 및 회차별 파일 저장 (Selenium 사용)
def crawl_all(webtoon_id, start_ep, end_ep, week, delay=2):
    """
    여러 웹툰 회차의 댓글을 크롤링하고 각 회차별로 별도의 CSV 파일로 저장합니다.
    """
    print("▶ 크롬 웹드라이버를 준비하는 중입니다...")

    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        for ep in range(start_ep, end_ep + 1):
            print(f"\n▶ 크롤링 시작: {ep}화")
            comments = crawl_comments(webtoon_id, ep, week, driver)

            if comments:
                os.makedirs("activeLearning", exist_ok=True)
                df = pd.DataFrame(comments)
                output_filename = f"activeLearning/comments_with_likes_no_label_{ep}.csv"

                # DataFrame을 CSV 파일로 저장하는 부분
                df.to_csv(output_filename, index=False, encoding="utf-8-sig")

                print(f"  - {ep}화 완료. '{output_filename}'에 {len(comments)}개 댓글 저장")
            else:
                print(f"  - {ep}화 댓글이 없습니다. 다음 회차로 넘어갑니다.")

            time.sleep(delay)
    finally:
        driver.quit()
        print("\n✅ 크롤링 작업이 완료되었습니다.")


# ✅ 실행 예시
if __name__ == "__main__":
    crawl_all(webtoon_id=750826, start_ep=50, end_ep=80, week="finish")

