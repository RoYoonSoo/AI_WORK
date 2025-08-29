import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options  # ✨ 옵션 임포트
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ✅ 한 회차 댓글 크롤링 (Selenium 사용)
def crawl_comments(webtoon_id, episode_no, week, driver):
    """
    특정 웹툰 회차의 댓글, 좋아요, 싫어요 수를 크롤링합니다.
    """
    url = f"https://comic.naver.com/webtoon/detail?titleId={webtoon_id}&no={episode_no}&week={week}"
    driver.get(url)  # 웹드라이버로 URL에 접속

    try:
        # try:
        #     all_comments_button = WebDriverWait(driver, 2).until(
        #         EC.element_to_be_clickable((By.XPATH, "//button[contains(., '전체 댓글')]"))
        #     )
        #     all_comments_button.click()
        #     time.sleep(1)
        # except Exception:
        #     print(f"  - {episode_no}화 '전체 댓글' 버튼을 찾지 못했지만 계속 진행합니다.")

        try:
            # 댓글 영역이 로딩될 때까지 최대 10초 대기
            # 'p' 태그 중 '_text_mfm2s_16' 클래스를 가진 요소가 나타날 때까지 기다립니다.
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, '_text_mfm2s_16'))
            )
            time.sleep(1)  # 로딩 후 여유 시간 추가
        except Exception:
            print(f"  - {episode_no}화 '전체 댓글' 버튼을 찾지 못했지만 계속 진행합니다.")

        # --- 모든 댓글이 로드될 때까지 스크롤 ---
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # 스크롤을 맨 아래로 내립니다.
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # 새 댓글이 로드될 시간을 줍니다.
            time.sleep(1.5)
            # 새 높이를 계산하고 이전 높이와 비교합니다.
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # 더 이상 스크롤이 되지 않으면 반복을 멈춥니다.
            last_height = new_height

    except Exception as e:
        print(f"❌ {episode_no}화 댓글 로딩 실패: {e}")
        driver.switch_to.default_content()
        return []

    # 로드된 페이지 소스를 가져와 BeautifulSoup으로 분석
    soup = BeautifulSoup(driver.page_source, "html.parser")
    comment_list_items = soup.find_all('p' , class_='_text_mfm2s_16')
    print(len(comment_list_items))
    comments_data = []
    for li in comment_list_items:
        # 댓글 본문
        comment = li.get_text(strip=True)

        parent_li = li.find_parent('li', class_='_root_1koau_1')
        # li 안의 모든 '좋아요/싫어요' div를 찾음
        # 이 리스트의 첫 번째 요소가 좋아요, 두 번째 요소가 싫어요 div입니다.
        print(parent_li)
        reaction_divs = parent_li.find_all('div', class_='_inside_2v7c9_21')
        print(len(reaction_divs))
        likes = "0"
        dislikes = "0"

        # 좋아요 숫자 가져오기
        # 첫 번째 div 요소가 존재하고, 그 안에 두 번째 span이 있다면
        if len(reaction_divs) >= 1:
            like_spans = reaction_divs[1].find_all('span')
            if len(like_spans) >= 2:
                likes = like_spans[1].get_text(strip=True)

        # 싫어요 숫자 가져오기
        # 두 번째 div 요소가 존재하고, 그 안에 두 번째 span이 있다면
        if len(reaction_divs) >= 2:
            dislike_spans = reaction_divs[2].find_all('span')
            if len(dislike_spans) >= 2:
                dislikes = dislike_spans[1].get_text(strip=True)

        # 'replace()'를 사용해 '[BEST]'를 ''(빈 문자열)로 바꿉니다.
        comment = comment.replace("BEST", "")

        comments_data.append({
            "comment": comment,
            "likes": likes,
            "dislikes": dislikes
        })

    # 원래 페이지로 컨텍스트 복귀
    driver.switch_to.default_content()

    return comments_data


# ✅ 여러 회차 크롤링 및 회차별 파일 저장 (Selenium 사용)
def crawl_all(webtoon_id, start_ep, end_ep, week, delay=2):
    """
    여러 웹툰 회차의 댓글을 크롤링하고 각 회차별로 별도의 CSV 파일로 저장합니다.
    """
    print("▶ 크롬 웹드라이버를 준비하는 중입니다...")

    # --- 크롬 옵션 설정 ---
    chrome_options = Options()
    # ✨수정된 부분: headless 옵션을 주석 처리하여 브라우저가 보이도록 설정
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    service = Service(ChromeDriverManager().install())
    # 설정한 옵션을 적용하여 드라이버 실행
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        for ep in range(start_ep, end_ep + 1):
            print(f"\n▶ 크롤링 시작: {ep}화")
            comments = crawl_comments(webtoon_id, ep, week, driver)

            if comments:
                os.makedirs("data", exist_ok=True)
                df = pd.DataFrame(comments)
                output_filename = f"data/comments_with_likes_{ep}.csv"
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
    # 웹툰 ID: 703846 (여신강림)
    # 시작 회차: 1, 끝 회차: 3
    # 요일: "wed" (수요일)
    crawl_all(webtoon_id=703846, start_ep=1, end_ep=10, week="wed")
