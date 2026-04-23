import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. HDR 이미지 로드
# cv2.IMREAD_UNCHANGED 플래그를 사용해야 데이터를 변환하지 않고 원본(float32 등) 그대로 읽습니다.
file_path = 'input.hdr'  # .hdr 또는 .exr 파일 경로
hdr_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

if hdr_img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하십시오.")
else:
    # 2. 데이터를 1차원 배열로 평탄화 (모든 채널 통합)
    pixels = hdr_img.ravel()

    # 3. 히스토그램 시각화
    plt.figure(figsize=(10, 6))
    
    # bins 숫자는 데이터의 정밀도에 따라 조절 가능합니다.
    # HDR 특성상 특정 구간에 값이 몰려 있을 수 있으므로 y축 로그 스케일을 권장합니다.
    plt.hist(pixels, bins=1000,range=(0,300000), color='blue', alpha=0.7, edgecolor='black')
    
    plt.yscale('log')  # 빈도수 차이가 클 경우 분포를 더 잘 보기 위해 로그 스케일 적용
    plt.title('Raw HDR Pixel Value Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.show()