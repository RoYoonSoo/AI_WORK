import pandas as pd
import os
import glob


def combine_csv_files(source_folder, output_filename):
    """
    지정된 폴더의 모든 CSV 파일의 내용을 하나의 CSV 파일로 합칩니다.
    'source_file' 열을 추가하여 각 행의 출처를 표시합니다.
    """
    # 소스 폴더가 존재하는지 확인
    if not os.path.isdir(source_folder):
        print(f"오류: '{source_folder}' 폴더를 찾을 수 없습니다.")
        print("CSV 파일이 들어있는 폴더를 스크립트와 같은 경로에 생성해주세요.")
        return

    # 소스 폴더에서 모든 CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(source_folder, '*.csv'))

    if not csv_files:
        print(f"'{source_folder}' 폴더에서 CSV 파일을 찾을 수 없습니다.")
        return

    # 모든 CSV 파일의 데이터프레임을 저장할 리스트
    all_dataframes = []

    print(f"총 {len(csv_files)}개의 CSV 파일을 '{output_filename}' 파일로 합치는 중입니다...")
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            # CSV 파일을 데이터프레임으로 읽기
            # 인코딩 문제 발생 시 'cp949'로 다시 시도
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                print(f"  - '{filename}' 파일 utf-8 디코딩 실패. cp949로 다시 시도합니다.")
                df = pd.read_csv(file_path, encoding='cp949')

            # 'source_file' 열을 추가하여 원본 파일명 저장
            df['source_file'] = filename

            all_dataframes.append(df)
            print(f"  - '{filename}' 파일 처리 완료.")

        except Exception as e:
            print(f"  - '{filename}' 파일 처리 중 오류 발생: {e}")

    if not all_dataframes:
        print("합칠 데이터가 없습니다. 파일 내용을 확인해주세요.")
        return

    # 모든 데이터프레임을 하나로 합치기
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # 합쳐진 데이터프레임을 새 CSV 파일로 저장 (인덱스 제외)
    # 엑셀에서 한글이 깨지지 않도록 'utf-8-sig' 인코딩 사용
    combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\n✅ 작업 완료! 모든 파일이 '{output_filename}'에 저장되었습니다.")


if __name__ == "__main__":
    # CSV 파일들이 저장된 폴더 이름
    SOURCE_FOLDER = 'data'

    # 최종적으로 합쳐질 파일 이름
    OUTPUT_FILENAME = 'combined_files.csv'

    combine_csv_files(SOURCE_FOLDER, OUTPUT_FILENAME)