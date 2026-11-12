# Day 3. 영화리뷰 텍스트 감성분석하기 — NSMC Sentiment Analysis

## 1. 프로젝트 개요

본 프로젝트에서는 네이버 영화 리뷰 데이터셋(NSMC)을 활용하여 한국어 텍스트의 감성분석(Sentiment Analysis) 모델을 구현하고 성능을 비교하였다.

MeCab 형태소 분석을 이용하여 텍스트를 전처리한 후, Mean Pooling, 1-D CNN, Bidirectional GRU(BiGRU)의 세 가지 딥러닝 모델을 구성하였다.

추가로 Word2Vec으로 초기화한 Embedding을 BiGRU에 적용하여 성능 변화를 확인하고, 학습 곡선, 단어 벡터 유사도, Confusion Matrix 및 오분류 사례를 분석하였다.

### 주요 개발 환경

- Environment: Google Colab
- Language: Python
- Deep Learning: PyTorch
- NLP: MeCab, Gensim Word2Vec
- Data Analysis: NumPy, Pandas, Matplotlib
- Evaluation: Scikit-learn
- Dataset: Naver Sentiment Movie Corpus (NSMC)

---

## 2. 프로젝트 목표

본 프로젝트의 주요 목표는 다음과 같다.

1. 한국어 영화 리뷰 데이터를 전처리하고 형태소 기반 단어사전을 구축한다.
2. 텍스트 데이터를 정수 ID와 Embedding Vector로 변환한다.
3. 서로 다른 구조의 딥러닝 모델 3개 이상을 구현하고 학습한다.
4. Accuracy, F1-score 및 학습 곡선을 통해 모델의 성능을 비교한다.
5. 학습된 Embedding Layer의 단어 벡터 특성을 분석한다.
6. Word2Vec 기반 Embedding 초기화에 따른 성능 변화를 검증한다.
7. Confusion Matrix 및 오분류 사례를 통해 모델의 한계를 분석한다.

---

## 3. 데이터셋 및 실험 환경

### 3.1. 데이터셋

본 프로젝트에서는 네이버 영화 리뷰 데이터셋인 NSMC를 사용하였다.

영화 리뷰는 긍정과 부정으로 구분되며, 레이블은 다음과 같이 구성된다.

| Label | 의미 |
|---|---|
| 0 | 부정적인 영화 리뷰 |
| 1 | 긍정적인 영화 리뷰 |

원본 NSMC는 학습 데이터 약 150,000건과 테스트 데이터 약 50,000건으로 구성된다.

이번 실험에서는 코드의 정상 동작 및 모델별 비교를 위해 일부 데이터를 사용하였다.

| 구분 | 사용 데이터 |
|---|---:|
| Training | 4,000 |
| Validation | 1,000 |
| Test | 2,000 |

긍정과 부정 클래스의 비율을 균형 있게 구성하였으며, 데이터 분할의 재현성을 위해 Random Seed 42를 사용하였다.

### 3.2. 공통 학습 조건

| Parameter | Setting |
|---|---|
| Vocabulary Size | 최대 10,000 |
| Embedding Dimension | 100 |
| Batch Size | 128 |
| Epoch | 4 |
| Optimizer | Adam |
| Learning Rate | 0.002 |
| Loss Function | CrossEntropyLoss |
| Model Selection | Best Validation F1-score |
| Random Seed | 42 |

최대 시퀀스 길이는 학습 데이터의 토큰 길이 분포를 분석한 뒤 결정하였다.

모델별 학습 조건과 데이터 분할을 동일하게 설정하여 신경망 구조에 따른 성능 차이를 비교하였다.

---

## 4. 프로젝트 수행 과정

### Step 1. 데이터 준비 및 탐색적 분석

NSMC 데이터셋을 불러온 후 데이터의 특성을 분석하였다.

주요 분석 항목은 다음과 같다.

- Training/Test 데이터 크기 확인
- 결측값 및 중복 리뷰 확인
- 긍정·부정 클래스 분포 확인
- 리뷰의 문자 길이 분포 시각화

이를 통해 데이터의 기본적인 특성을 파악하고 이후 전처리 및 학습에 필요한 조건을 확인하였다.

### Step 2. 한국어 형태소 분석 및 단어사전 구축

MeCab을 이용하여 리뷰를 형태소 단위로 분리하였다.

이후 빈도 기반 단어사전을 생성하고, 각 형태소를 정수 ID로 변환하였다.

특수 토큰을 구분하여 다음과 같이 정의하였다.

| Token | ID | Purpose |
|---|---:|---|
| `<PAD>` | 0 | Padding |
| `<BOS>` | 1 | 문장 시작 |
| `<UNK>` | 2 | 미등록 단어 |
| `<EOS>` | 3 | 문장 종료 |

감성분석에 중요한 부정 표현이 제거되지 않도록 불용어 목록을 조정하였다.

또한 Encoding 및 Decoding을 수행하여 정수 ID 변환 과정을 검증하였다.

### Step 3. 시퀀스 길이 분석 및 Padding

문장별 토큰 길이를 분석하고 평균, 중앙값, 95백분위수 및 최대 길이를 확인하였다.

학습 데이터의 길이 분포를 바탕으로 최대 시퀀스 길이를 설정하고, 길이가 다른 문장에 Padding을 적용하였다.

이후 모든 데이터를 PyTorch Tensor로 변환하여 신경망에 입력할 수 있도록 구성하였다.

### Step 4. 세 가지 감성분류 모델 구현

동일한 입력 데이터와 단어사전을 기반으로 다음 세 가지 모델을 구현하였다.

| Model | Architecture | 특징 |
|---|---|---|
| Model 1 | Embedding + Mean Pooling | 단어 벡터의 평균적인 특징 활용 |
| Model 2 | Embedding + 1-D CNN | 인접한 토큰의 지역적 패턴 추출 |
| Model 3 | Embedding + BiGRU | 순차적 문맥과 양방향 정보 활용 |

각 모델의 출력은 부정과 긍정에 대한 두 개의 분류 점수로 구성하였다.

### Step 5. 모델 학습 및 평가

세 가지 모델을 동일한 조건으로 학습하고 Epoch별 Training Loss, Validation Loss, Accuracy 및 F1-score를 기록하였다.

최종 Epoch의 모델을 무조건 선택하는 대신, Validation F1-score가 가장 높은 시점의 모델을 저장하여 최종 평가에 사용하였다.

### Step 6. Loss 및 Accuracy 시각화

모델별 Training/Validation Loss와 Accuracy를 시각화하였다.

이를 통해 학습 진행에 따른 성능 변화를 확인하고, 과적합 여부를 분석하였다.

### Step 7. Embedding Layer 분석

학습이 완료된 BiGRU 모델의 Embedding Matrix를 추출하였다.

단어별 벡터를 확인하고, Cosine Similarity를 이용하여 특정 단어와 벡터 공간에서 유사한 단어들을 검색하였다.

이 과정에서 Embedding이 단어의 고정된 의미를 직접 표현하는 것이 아니라, 모델의 학습 목적에 따라 조정되는 파라미터라는 점을 확인하였다.

### Step 8. Word2Vec 적용 및 추가 학습

기본 BiGRU의 Embedding 초기화 방식에 따른 성능 변화를 검증하기 위해 Word2Vec을 적용하였다.

이번 실험에서는 외부에서 제공되는 대규모 사전학습 모델 대신, NSMC Training 데이터로 Word2Vec을 직접 학습하였다.

이후 Word2Vec 벡터를 BiGRU의 Embedding Layer 초기값으로 적용하고, 동일한 학습 조건에서 모델을 재학습하였다.

**Word2Vec 분석 결과**

| 항목 | 결과 |
|---|---:|
| Word2Vec Vector Dimension | 100 |
| Word2Vec 학습 단어 수 | 3,645 |
| 전체 일반 단어사전 수 | 8,388 |
| Vocabulary Coverage | 43.45% |

Word2Vec에 등록되지 않은 단어는 별도의 초기값을 사용하도록 처리하였다.

---

## 5. 모델 성능 비교 결과

### 5.1. 최종 테스트 성능

| Model | Test Accuracy | Test F1-score |
|---|---:|---:|
| Mean Pooling | 72.65% | 0.6976 |
| 1-D CNN | 75.50% | 0.7569 |
| BiGRU | 72.60% | 0.7204 |
| **BiGRU + Word2Vec** | **77.15%** | **0.7920** |

### 5.2. 주요 결과 해석

**1. 모델 구조에 따른 성능 차이**

기본 모델 세 가지 중에서는 1-D CNN이 가장 높은 Test Accuracy와 F1-score를 기록하였다.

반면 BiGRU는 순차적인 문맥을 학습할 수 있는 모델임에도 불구하고 1-D CNN보다 낮은 성능을 보였다.

이는 모델 구조가 복잡하다고 해서 항상 분류 성능이 향상되는 것은 아님을 보여준다.

**2. Word2Vec 적용에 따른 성능 변화**

기본 BiGRU의 Test F1-score는 0.7204였으나, Word2Vec 초기화를 적용한 BiGRU에서는 0.7920으로 향상되었다.

F1-score 기준 약 0.0716의 절대적인 상승을 확인하였다.

다만 Word2Vec 자체의 효과를 일반화하려면 복수 Seed 및 다른 학습 조건에서 추가 검증이 필요하다.

**3. Embedding의 중요성**

동일한 BiGRU 모델 구조에서도 Embedding 초기화 조건을 변경했을 때 성능 차이가 발생하였다.

이를 통해 입력 단어를 벡터로 표현하는 방식이 최종 감성분류 성능에 영향을 줄 수 있음을 확인하였다.

---

## 6. 학습 곡선 및 과적합 분석

Training/Validation Loss와 Accuracy를 비교한 결과, 모델별로 학습 특성에 차이가 나타났다.

- **Mean Pooling:** Training과 Validation Loss가 전반적으로 감소하였으며, 추가 학습의 여지가 관찰되었다.
- **1-D CNN:** Validation Loss는 3 Epoch까지 감소한 후 4 Epoch에서 소폭 상승하였다.
- **BiGRU:** Training Loss는 지속적으로 감소했지만 Validation Loss는 2 Epoch 이후 상승하였다.
- **BiGRU + Word2Vec:** Validation F1-score는 3 Epoch에서 가장 높았으며, 이후 Validation Loss 상승이 확인되었다.

일부 모델에서 과적합 징후가 관찰되었으므로, 단순히 Epoch를 증가시키기보다 검증 성능을 기준으로 학습 결과를 평가하는 것이 중요함을 확인하였다.

---

## 7. Confusion Matrix 및 오분류 분석

Validation F1-score를 기준으로 선택한 최고 성능 모델인 BiGRU + Word2Vec의 Confusion Matrix를 분석하였다.

| 실제 정답 | 부정 예측 | 긍정 예측 |
|---|---:|---:|
| 부정 | 673 | 327 |
| 긍정 | 130 | 870 |

전체 테스트 리뷰 2,000건 중 1,543건을 올바르게 분류하였다.

부정 리뷰를 긍정으로 잘못 분류한 경우가 327건으로, 긍정 리뷰를 부정으로 잘못 분류한 경우인 130건보다 많았다.

따라서 본 모델은 긍정 리뷰를 상대적으로 잘 찾아내지만 부정 리뷰를 긍정으로 분류하는 오류가 비교적 많이 발생하는 특성을 보였다.

추가로 실제 오분류 리뷰를 출력하여 모델이 잘못 판단한 사례를 검토하였다.

이러한 분석을 통해 단순 Accuracy뿐 아니라 예측 오류의 방향과 유형도 확인할 필요가 있음을 알 수 있었다.

---

## 8. 오류 해결 및 추가 구현

실습 과정에서 발생할 수 있는 오류를 해결하고, 원본 프로젝트에서 요구한 내용을 확장하였다.

| 항목 | 개선 내용 |
|---|---|
| MeCab 설치 및 호환성 | Colab 환경에서 `mecab-ko` 활용 |
| 특수 토큰 처리 | PAD, BOS, UNK, EOS ID 구분 |
| 데이터 전처리 | 결측값, 중복 리뷰 처리 및 데이터 분할 |
| Padding | 토큰 길이 분석에 기반한 최대 길이 설정 |
| 모델 비교 | Mean Pooling, CNN, BiGRU 구현 |
| 학습 안정화 | Gradient Clipping 및 Validation 기반 모델 선택 |
| Embedding 분석 | Cosine Similarity 기반 단어 벡터 비교 |
| Word2Vec 적용 | Embedding Matrix 구축 및 Vocabulary Coverage 계산 |
| 추가 성능 분석 | F1-score, Confusion Matrix, 오분류 사례 |

---

## 9. 프로젝트 회고

### 9.1. 배운 점

이전 프로젝트에서는 텍스트를 정수 ID로 변환하는 토큰화 및 단어사전 구축 과정에 집중하였다.

이번 프로젝트에서는 정수로 변환된 데이터가 Embedding Layer를 통해 벡터로 표현되고, 신경망 모델에서 학습되어 최종 감성분류 결과를 생성하는 전체 과정을 이해할 수 있었다.

특히 신경망 구조뿐 아니라 Embedding의 초기화 방식과 학습 데이터의 특성이 모델 성능에 영향을 줄 수 있음을 확인하였다.

### 9.2. 깨달은 점

**첫째, 모델의 복잡도가 반드시 우수한 성능을 보장하지는 않는다.**

기본 BiGRU보다 1-D CNN의 성능이 높게 나타난 결과를 통해 모델 구조는 실제 데이터 특성과 검증 결과를 바탕으로 선택해야 함을 알 수 있었다.

**둘째, 학습 성능과 일반화 성능은 다르다.**

Training Loss가 감소하더라도 Validation Loss가 증가할 수 있으며, 과적합이 발생하면 새로운 데이터에 대한 예측 성능이 제한될 수 있음을 확인하였다.

**셋째, 모델을 평가할 때는 단일 지표만으로 판단해서는 안 된다.**

Accuracy와 F1-score 외에도 Confusion Matrix 및 오분류 사례를 함께 분석하면 모델의 성능과 한계를 더욱 구체적으로 파악할 수 있음을 배웠다.

### 9.3. 아쉬운 점

이번 프로젝트에서는 전체 NSMC 데이터 대신 일부 데이터만 활용하였고, 4 Epoch의 제한된 학습을 진행하였다.

또한 외부 대규모 사전학습 Word2Vec 모델을 사용하지 못하고 NSMC 학습 데이터로 자체 Word2Vec을 구축하여 실험하였다.

따라서 사전학습 모델 활용에 따른 일반적인 성능 개선 효과를 확인했다고 보기는 어렵다.

### 9.4. 개선 방향

1. 전체 NSMC 학습 데이터 활용 및 학습 Epoch 확대
2. Early Stopping과 Dropout 등 정규화 기법 개선
3. 여러 Random Seed를 적용한 반복 실험
4. 외부 사전학습 Word2Vec과 자체 학습 Word2Vec 비교
5. 부정 표현, 반어법 및 문맥 부족에 따른 오분류 유형 분석
6. 모델별 학습 시간과 파라미터 수를 함께 고려한 효율성 비교

---

## 10. 최종 결론

본 프로젝트에서는 한국어 영화 리뷰를 대상으로 전처리, 형태소 분석, Embedding, 신경망 모델 구성, 학습 및 평가에 이르는 감성분류의 전체 과정을 구현하였다.

세 가지 기본 모델을 비교한 결과 1-D CNN이 가장 높은 성능을 나타냈으며, Word2Vec을 적용한 BiGRU가 Test Accuracy 77.15%, F1-score 0.7920으로 전체 실험 중 가장 높은 성능을 기록하였다.

이를 통해 **모델 구조와 단어 표현 방식 모두 감성분류 성능에 영향을 미칠 수 있으며, 모델의 성능을 정확하게 이해하기 위해서는 학습 곡선과 오분류 사례를 포함한 종합적인 평가가 필요하다**는 점을 확인하였다.

향후 데이터 규모를 확장하고 다양한 임베딩 및 학습 조건을 비교한다면 모델의 일반화 성능과 실험 결과의 신뢰성을 더욱 높일 수 있을 것으로 기대한다.

---

## 11. 참고 자료

- [NSMC — Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [KoNLPy Documentation](https://konlpy.org/)

---

**본 프로젝트는 Google Colab에서 수행하였으며, 코드 및 실행 결과는 상단의 Jupyter Notebook 파일에서 확인할 수 있다.**
