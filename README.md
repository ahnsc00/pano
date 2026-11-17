# PANO

파레토 보상 기반 선택·갱신 알고리즘.

## 실행

Python 3.12 기준.

```bash
pip install -r requirements.txt
python reproduce.py --scenarios synthetic
```

파레토 분포에서 생성한 보상으로 PANO와 CRTM 성능 비교.
20개 seed에서 각각 100만 회 실행 후, 평균 누적 regret 그래프.
그래프: `results/paper/figures/` 저장. 수치: `results/paper/` 저장.

## 파일

- `pano.py`: 핵심 알고리즘.
- `xue_crtm.py`: 비교 알고리즘.
- `reproduce.py`: 설정·실험·그래프 생성.

## 데이터

- [freMTPL2](https://dutangc.github.io/CASdatasets/reference/freMTPL.html): 프랑스 자동차 책임보험 계약·청구 자료.
- [Health insurance portfolio](https://data.mendeley.com/datasets/386vmj2tbk/4): 2017–2019년 건강보험 포트폴리오 자료.
