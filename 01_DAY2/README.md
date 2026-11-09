# Day 2. 멋진 단어사전 만들기 — SentencePiece & NSMC

## 1. 프로젝트 개요

본 프로젝트에서는 자연어처리(NLP)의 기본 단계인 토큰화(Tokenization)와 단어사전(Vocabulary) 구축 과정을 학습하고, 공백 기반 토큰화, MeCab 형태소 분석, SentencePiece 서브워드 토큰화 방법을 구현하였다.

이후 네이버 영화 리뷰 데이터셋(NSMC)에 각 토크나이저를 적용하여 동일한 BiGRU 감성분류 모델에서 성능을 비교하였다. 추가로 SentencePiece의 토큰화 알고리즘과 Vocabulary 크기를 변경하여 성능에 미치는 영향을 분석하였다.

**실습 Notebook:** [02_SentencePiece_NSMC.ipynb](./02_SentencePiece_NSMC.ipynb)

**사용 환경 및 기술**

- Environment: Google Colab, Python, PyTorch
- Tokenization: Whitespace, MeCab, SentencePiece
- Model: Bidirectional GRU
- Dataset: Korean-English Parallel Corpus, Naver Sentiment Movie Corpus (NSMC)
- Evaluation: Accuracy, F1-score

---

## 2. 프로젝트 수행 과정

### Step 1. 텍스트 전처리 및 공백 기반 토큰화

한국어 말뭉치를 불러와 중복 문장을 제거하고, 문장 길이 분포를 분석하였다.

길이가 10자 이상 150자 미만인 문장을 선택하여 `filtered_corpus`를 구성하고, `split()`을 사용하여 공백 기반 토큰화를 수행하였다.

Tokenizer를 이용해 단어별 정수 ID를 생성하고 Padding을 적용하여 Tensor 데이터를 구축하였다.

### Step 2. MeCab 형태소 기반 토큰화

MeCab을 사용하여 한국어 문장을 형태소 단위로 분리하고, 형태소 기반 단어사전과 Tensor를 생성하였다.

`sequences_to_texts()` 및 `index_word`를 이용한 Decoding을 수행하여 정수 ID로 변환된 데이터를 다시 토큰으로 복원하였다.

실행 과정에서 Colab의 Python 환경과 MeCab 라이브러리 호환성 문제를 확인하였으며, `mecab-ko` 패키지를 활용하여 형태소 분석을 구현하였다.

### Step 3. SentencePiece 단어사전 학습 및 Tokenizer 구현

전처리한 한국어 말뭉치를 활용하여 SentencePiece Unigram 모델을 학습하였다.

목표 Vocabulary 크기를 8,000으로 설정하였으며, 학습 결과로 생성된 `.model` 및 `.vocab` 파일을 이용해 문장을 Encoding하고 Decoding하였다.

이후 `sp_tokenize()` 함수를 구현하여 문장별 정수 ID 변환, 단어사전 구축, Padding 및 Tensor 생성을 수행하였다.

특히 Padding ID와 Unknown Token ID를 구분하고, PyTorch의 `pad_sequence()`에 전달하는 데이터를 Tensor로 변환하여 안정적으로 처리하였다.

### Step 4. 네이버 영화 리뷰 감성분석 적용

NSMC 데이터셋의 긍정·부정 리뷰에 MeCab 및 SentencePiece를 적용하고, 동일한 BiGRU 분류 모델을 학습하여 토크나이저별 성능을 비교하였다.

Sentiment Classification은 부정 리뷰를 0, 긍정 리뷰를 1로 분류하는 이진 분류 문제로 구성하였다.

**실험 조건**

| 항목 | 설정 |
|---|---|
| Training Data | 8,000건 |
| Validation Data | 2,000건 |
| Test Data | 2,000건 |
| Classification Model | Bidirectional GRU |
| Epoch | 3 |
| Batch Size | 128 |
| Embedding Dimension | 64 |
| Hidden Size | 64 |
| Maximum Sequence Length | 64 |
| Optimizer | Adam |
| Learning Rate | 0.002 |
| Evaluation | Accuracy, F1-score |

비교 실험에서는 다음 네 가지 토큰화 조건을 사용하였다.

| Case | Tokenizer | Vocabulary 목표 크기 |
|---|---|---:|
| Baseline | MeCab | 8,000 |
| Case 1 | SentencePiece Unigram | 8,000 |
| Case 2 | SentencePiece BPE | 8,000 |
| Case 3 | SentencePiece Unigram | 4,000 |

SentencePiece 단어사전은 NSMC의 학습 데이터만 사용하여 새롭게 학습하였다. 각 조건에서 동일한 데이터 분할과 신경망 구조를 적용하였으며, Validation F1-score를 기준으로 학습 중 가장 성능이 좋은 Epoch의 모델을 선택하였다.

---

## 3. 주요 실험 결과

### 토크나이저별 F1-score 비교

| Tokenizer | Test F1-score (약) |
|---|---:|
| MeCab 8,000 | 0.77 |
| SentencePiece Unigram 8,000 | 0.69 |
| SentencePiece BPE 8,000 | 0.71 |
| SentencePiece Unigram 4,000 | 0.73 |

*수치는 그래프에서 확인한 근삿값이며, 정확한 Accuracy 및 F1-score 결과는 실습 Notebook의 최종 비교표를 참고할 수 있다.*

### 결과 분석

**1. MeCab과 SentencePiece 비교**

MeCab 기반 형태소 토큰화가 약 0.77의 F1-score로 가장 높은 성능을 나타냈다. 이는 본 실험의 데이터 규모와 BiGRU 학습 조건에서 형태소 단위 분리가 유리하게 작용했을 가능성을 보여준다.

**2. Unigram과 BPE 비교**

동일한 Vocabulary 목표 크기 8,000 조건에서 BPE가 Unigram보다 높은 F1-score를 기록하였다. 이를 통해 서브워드 분할 알고리즘의 차이가 감성분류 성능에 영향을 줄 수 있음을 확인하였다.

**3. Vocabulary 크기에 따른 성능 변화**

Unigram의 Vocabulary 목표 크기를 8,000에서 4,000으로 줄였을 때 F1-score가 향상되었다. 따라서 단어사전의 크기를 증가시키는 것이 반드시 분류 성능의 향상으로 이어지지는 않음을 확인하였다.

**4. 실험의 한계**

본 실험은 일부 데이터셋과 3 Epoch를 이용한 단일 학습 결과이므로, 토크나이저 간의 일반적인 성능 우열로 단정하기 어렵다. 또한 실제 Vocabulary 크기, 토큰 길이 분포, Truncation 비율 등의 차이가 결과에 영향을 미쳤을 수 있다.

---

## 4. 오류 해결 및 추가 구현

실습 과정에서 발생한 주요 오류를 다음과 같이 해결하였다.

| 문제 | 원인 | 해결 방법 |
|---|---|---|
| `NameError` | 이전 셀의 변수 미정의 | 전처리 셀 순서대로 실행 |
| MeCab Import 오류 | 패키지 미설치 및 실행 환경 차이 | `mecab-ko` 설치 및 사용 |
| MeCab 설치 실패 | Python 버전 및 패키지 빌드 호환성 | 호환되는 패키지로 교체 |
| `pad_sequence()` 입력 오류 가능성 | Python 리스트와 Tensor 자료형 차이 | `torch.tensor()` 변환 |
| Padding 및 UNK 혼동 | 특수 토큰 ID 중복 | PAD/UNK ID 분리 |

추가로 단어사전 크기 비교, Encoding/Decoding 검증, 네 가지 토크나이저 조건에서의 감성분류 성능 비교를 수행하였다.

---

## 5. 회고 및 개선 방향

### 배운 점

문장을 토큰으로 분할하는 방식에 따라 단어사전의 구성과 Tensor의 형태가 달라진다는 것을 확인하였다. 특히 SentencePiece 모델을 직접 학습하고 Encoding/Decoding을 구현하면서 서브워드 기반 토큰화의 작동 과정을 이해할 수 있었다.

또한 동일한 BiGRU 모델이라도 입력 텍스트의 토큰화 방식 및 Vocabulary 설정에 따라 분류 성능이 달라질 수 있음을 실험적으로 확인하였다.

### 아쉬운 점

학습 데이터와 Epoch 수가 제한적이었으며, 단일 Random Seed를 기반으로 실험하여 결과의 재현성을 충분히 평가하지 못하였다.

### 향후 개선 방향

전체 NSMC 데이터 사용, Epoch 확대 및 복수 Random Seed 실험을 통해 결과의 신뢰성을 높이고자 한다. 추가로 시퀀스 길이, Truncation 비율 및 미등록 토큰 비율을 분석하여 토큰화 방식이 감성분류 성능에 미치는 영향을 더욱 구체적으로 검토할 필요가 있다.

---

## 6. 실행 및 참고 자료

실습 Notebook은 Google Colab 환경에서 실행하였다. 코드 재실행 시 필요한 라이브러리를 설치한 후 데이터 전처리, 단어사전 학습, Tensor 생성, 감성분류 모델 학습 순서로 실행하면 된다.

SentencePiece 학습 과정에서 생성되는 `.model`, `.vocab` 파일 및 실험 결과 CSV는 코드 실행으로 다시 생성할 수 있다. Notebook에는 코드와 주요 실행 결과를 포함하였다.

**참고 자료**

- [SentencePiece — Google](https://github.com/google/sentencepiece)
- [KoNLPy Documentation](https://konlpy.org/)
- [NSMC — Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

*본 문서는 Day 2 NLP 프로젝트의 수행 과정 및 주요 실험 결과를 정리한 것이다.*
