# DAY 10. Transformer 기반 한국어 대화형 챗봇 구현 및 성능 분석

## 1. 프로젝트 개요

본 프로젝트는 기존 영어 기반 Transformer 챗봇 실습을 한국어 데이터에 적용하고, 데이터 전처리부터 토크나이저 학습, 모델 설계, 학습 및 추론까지 자연어처리의 전체 과정을 구현하는 것을 목적으로 한다.

기본 Transformer 모델 구현 이후에는 DeepSeek 특강에서 학습한 Attention 구조와 연산 최적화 기술을 적용하여 모델별 생성 성능, 메모리 사용량 및 추론 효율을 비교하였다. 추가적으로 다중 턴 대화, 단기 및 장기 기억, RAG, GRPO, 사전학습 언어 모델과 LoRA를 활용한 추가 학습을 수행하였다.

단순히 챗봇을 구현하는 것에서 그치지 않고, 각각의 기술이 실제 모델 성능에 어떠한 영향을 미치는지 확인하고 구현 과정에서 발생한 문제와 한계를 분석하고자 하였다.

### 프로젝트 실행 파일

[Korean_Transformer_Project_Submission_Final.ipynb](./Korean_Transformer_Project_Submission_Final.ipynb)

[Google Colab에서 실행하기](https://colab.research.google.com/github/clodspring/ERICA_X_MODULABS_NLP/blob/main/07_DAY10/Korean_Transformer_Project_Submission_Final.ipynb)

---

## 2. 개발 환경

| 구분 | 사용 환경 및 기술 |
|---|---|
| Language | Python |
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Deep Learning Framework | PyTorch |
| Tokenizer | SentencePiece BPE |
| Pretrained Model | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct |
| Fine-tuning | PEFT, LoRA |
| Database | SQLite |
| Retrieval | TF-IDF, Cosine Similarity |
| Interface | Gradio |
| Visualization | Matplotlib, Pandas |

---

## 3. 데이터 수집 및 전처리

송영숙님의 공개 한국어 챗봇 데이터셋을 사용하였다. 해당 데이터는 일상 대화, 이별 및 사랑과 관련된 질문과 답변으로 구성되어 있다.

데이터 출처: [songys/Chatbot_data](https://github.com/songys/Chatbot_data)

### 3.1. 데이터 구성

| 구분 | 데이터 수 |
|---|---:|
| 원본 데이터 | 11,823 |
| 전처리 완료 데이터 | 11,750 |
| Training Set | 9,403 |
| Validation Set | 1,174 |
| Test Set | 1,173 |

한국어의 특성을 고려하여 Unicode NFC 정규화, 제어 문자 제거, 공백 정리, 결측값 및 중복 문답 제거를 수행하였다. 한글뿐 아니라 숫자와 구두점도 보존하였으며, 동일한 질문이 학습 데이터와 평가 데이터에 중복되지 않도록 질문 단위로 데이터를 분리하였다.

### 3.2. SentencePiece Tokenization

한국어는 조사와 어미에 따라 단어의 형태가 다양하게 변화하므로 공백을 기준으로 단어를 분리하는 방식보다 서브워드 기반 토크나이징을 적용하였다.

| 항목 | 설정 |
|---|---|
| Tokenizer | SentencePiece |
| Tokenization Algorithm | BPE |
| Vocabulary Size | 4,000 |
| Character Coverage | 1.0 |
| Byte Fallback | True |
| Special Tokens | PAD, UNK, BOS, EOS |
| Role Tokens | SYSTEM, USER, ASSISTANT, MEMORY, DOC |

토크나이저는 Training Set만 사용하여 학습하였으며, 평가 과정에서 발생할 수 있는 어휘 손실을 줄이기 위해 Byte Fallback을 적용하였다.

최종적으로 자체 구현 모델의 평가 답변에서 UNK 토큰이 생성되지 않았음을 확인하였다.

---

## 4. Transformer 모델 설계

### 4.1. 기본 Transformer 구조

PyTorch를 이용하여 Encoder–Decoder Transformer 구조를 직접 구현하였다.

Encoder에서는 입력 질문의 문맥 정보를 추출하고, Decoder에서는 Masked Self-Attention과 Cross-Attention을 통해 이전 토큰 및 Encoder의 출력 정보를 활용하여 답변을 자기회귀 방식으로 생성하도록 구성하였다.

| Hyperparameter | 설정값 |
|---|---:|
| Vocabulary Size | 4,000 |
| Embedding Dimension | 128 |
| Attention Heads | 8 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Feed Forward Dimension | 384 |
| Batch Size | 32 |
| Learning Rate | 0.0008 |
| Maximum Epochs | 40 |
| Early Stopping Patience | 6 |
| Optimizer | AdamW |
| Scheduler | Warm-up + Cosine Decay |

기존 Transformer 구조를 기반으로 Pre-Layer Normalization, GELU, 가중치 공유 및 Decoder Self-Attention의 Rotary Position Embedding을 적용하였다.

### 4.2. Attention 구조 확장

Encoder와 Cross-Attention은 MHA 구조로 유지하고, Decoder Self-Attention 구조를 변경하여 각 Attention 방식의 성능을 비교하였다.

| Model | Full Name | 주요 특징 |
|---|---|---|
| MHA | Multi-Head Attention | 각 Query Head에 독립적인 Key와 Value 사용 |
| MQA | Multi-Query Attention | 모든 Query Head가 하나의 Key와 Value 공유 |
| GQA | Grouped-Query Attention | Query Head를 그룹으로 나누어 Key와 Value 공유 |
| MLA | Multi-head Latent Attention | Key와 Value를 저차원 Latent Representation으로 압축 |
| DSA | DeepSeek Sparse Attention | Indexer를 이용하여 관련 토큰을 선택적으로 참조 |

MLA에서는 Matrix Absorption을 적용하여 명시적인 Key와 Value 복원 방식과의 수치적 동등성을 검증하였다.

DSA는 DeepSeek-V3.2의 구조를 참고하여 Dense Attention 기반 Indexer 학습과 Sparse Attention을 구현하였다. 실제 DeepSeek 모델 전체를 재현한 것은 아니며, 핵심 연산 원리를 확인하기 위한 축소 구현이다.

---

## 5. 모델 학습 및 평가

### 5.1. 기본 Transformer 학습 결과

기본 MHA 모델은 총 1,303,552개의 파라미터로 구성하였다.

최대 40 Epoch로 학습을 설정하였으나, 10 Epoch 이후 Training Loss가 감소하는 동안 Validation Loss가 증가하는 과적합 현상이 나타났다. 이에 따라 Early Stopping을 적용하였으며, Validation Loss가 가장 낮은 10 Epoch의 모델을 최종 평가에 사용하였다.

| 평가 항목 | 결과 |
|---|---:|
| Best Epoch | 10 |
| Best Validation Loss | 3.3969 |
| Test Loss | 3.3827 |
| Token Accuracy | 45.41% |
| Character LCS F1 | 0.2572 |
| EOS Completion Rate | 99% |
| UNK Generation Rate | 0% |
| Empty Response Rate | 0% |
| Repeated Bigram Rate | 1.78% |

Test Loss와 Token Accuracy는 전체 테스트 데이터 1,173건을 대상으로 측정하였다. Character LCS F1 및 생성 관련 지표는 고정된 시드로 추출한 100개 문답을 대상으로 평가하였다.

### 5.2. Attention 모델별 성능 비교

| Model | Test Loss | Token Accuracy | Character LCS F1 |
|---|---:|---:|---:|
| MHA | 3.3827 | 45.41% | 0.2572 |
| MQA | 3.4401 | 43.34% | 0.2556 |
| GQA | 3.4075 | 44.34% | **0.2718** |
| MLA | 3.4374 | 44.38% | 0.2513 |
| MLA-control | 3.4603 | 44.61% | 0.2505 |
| DSA | 3.4615 | 44.54% | 0.2570 |
| Dialogue | 3.4144 | **45.99%** | 0.2454 |

MLA-control은 DSA의 추가 학습 효과를 구분하기 위한 Dense Attention 대조군이며, Dialogue는 MHA에 대화 및 기억 관련 데이터를 추가 학습한 모델이다.

GQA가 문자 유사도에서 가장 높은 값을 나타냈으며, Attention 구조를 변경하더라도 전체적인 생성 성능에는 큰 차이가 발생하지 않았다.

다만 Token Accuracy는 이전 정답 토큰이 제공된 상태에서의 예측 정확도이며, 실제 대화의 정답률과는 다르다. Character LCS F1 역시 의미가 동일하지만 표현이 다른 답변을 충분히 평가하지 못한다는 한계가 있다.

### 5.3. Decoding 방식 비교

동일한 모델을 대상으로 Greedy Search, Beam Search, Top-k Sampling, Top-p Sampling을 적용하였다. Temperature 변화와 Repetition Penalty, No-repeat Trigram 적용에 따른 답변의 다양성 및 반복 생성 현상도 비교하였다.

기본 MHA 모델에서 반복 제어를 적용한 결과는 다음과 같다.

| 평가 항목 | 기본 생성 | 반복 제어 적용 |
|---|---:|---:|
| Character LCS F1 | 0.2572 | 0.2636 |
| EOS Completion Rate | 99% | 100% |
| Repeated Bigram Rate | 1.78% | 0.11% |

반복 제어를 통해 동일한 토큰 조합이 반복되는 현상을 완화할 수 있었다. 그러나 반복률 감소가 질문에 대한 이해 능력이나 답변의 의미적 정확성 향상을 보장하지는 않았다.

---

## 6. KV Cache 및 Attention 연산 효율 분석

### 6.1. KV Cache 구현 및 검증

자기회귀 방식으로 토큰을 생성할 때 이전 토큰의 Key와 Value를 반복하여 계산하지 않도록 KV Cache를 구현하였다.

Cache 적용 전후의 Logits 및 생성된 토큰 ID를 비교하여 연산의 동등성을 검증하였으며, 동일한 Greedy Decoding 결과를 얻을 수 있음을 확인하였다.

### 6.2. Attention 구조별 Cache 저장량

32개 토큰, FP32, Decoder 2개 Layer를 기준으로 Self-Attention Cache의 저장량을 비교하였다.

| Model | Cache Size | MHA 대비 절감률 |
|---|---:|---:|
| MHA | 64 KiB | - |
| MQA | 8 KiB | 87.50% |
| GQA | 16 KiB | 75.00% |
| MLA | 18 KiB | 71.88% |
| DSA | 20 KiB | 68.75% |

각 Attention 구조에서 Key와 Value를 공유하거나 압축함에 따라 Cache 저장량이 감소하는 것을 확인하였다.

DSA는 선택적 Attention에 필요한 Indexer 정보를 추가로 저장하므로 MLA보다 Cache 저장량이 소폭 증가하였다.

다만 실제 GPU 추론에서는 Cache 저장량 감소가 동일한 비율의 속도 향상으로 이어지지 않았다. 작은 모델과 짧은 시퀀스에서는 추가적인 행렬 연산과 토큰 선택 비용도 고려해야 한다는 점을 확인하였다.

---

## 7. Long-context 및 DSA 성능 분석

긴 시퀀스에서 필요한 정보를 선택적으로 참조하는 DSA의 특성을 분석하기 위해 별도의 합성 Key-Value 검색 실험을 수행하였다.

학습 과정에서는 짧은 시퀀스부터 긴 시퀀스로 학습 범위를 확장하는 Curriculum Learning을 적용하였다.

| Sequence Length | Dense MLA Accuracy | Sparse MLA Accuracy |
|---:|---:|---:|
| 19 | 100.00% | 96.88% |
| 35 | 100.00% | 76.56% |
| 67 | 100.00% | 73.44% |
| 131 | 100.00% | 43.75% |
| 259 | 96.88% | 23.44% |
| 515 | 57.81% | 7.81% |

Dense MLA는 학습 범위보다 긴 시퀀스에서도 일정 수준의 정보 검색 능력을 확보하였다. 반면 Sparse MLA는 시퀀스 길이가 증가할수록 정확도가 크게 감소하였다.

이를 통해 Attention 대상 토큰의 수를 줄이는 것뿐 아니라, 필요한 토큰을 정확하게 선택할 수 있도록 Indexer를 충분히 학습하는 과정이 중요하다는 것을 확인하였다.

본 실험에서는 DSA의 구조적 동작을 확인하였으나 Dense Attention 대비 성능 유지 및 추론 속도 향상은 달성하지 못하였다. 또한 합성 토큰을 이용한 검색 과제이므로 실제 자연어 장문 이해 성능과는 구분하여 해석하였다.

---

## 8. 대화 능력 및 기억 시스템 확장

### 8.1. Multi-turn Dialogue

기존 단일 질문 및 답변 구조를 확장하여 이전 대화 내용을 현재 입력에 포함할 수 있도록 구성하였다.

역할 토큰을 이용하여 사용자와 챗봇의 발언을 구분하고, 이전 대화 내용과 상황 정보를 활용할 수 있도록 합성 데이터를 추가 학습하였다.

### 8.2. 단기 및 장기 기억

단기 기억은 이전 대화 기록을 현재 모델 입력에 포함하는 방식으로 구현하였다. 장기 기억은 SQLite를 이용하여 사용자별 정보를 저장하고, 필요한 정보를 검색하여 모델의 입력 문맥에 전달하도록 구성하였다.

기억의 저장, 조회, 수정 및 삭제 기능을 구현하였으며, 외부 데이터베이스의 동작과 모델 자체의 기억 활용 성능을 구분하여 평가하였다.

### 8.3. Persona 및 RAG

System Prompt를 이용하여 챗봇의 역할과 상황을 설정할 수 있도록 구성하였다.

RAG(Retrieval-Augmented Generation)는 TF-IDF와 Cosine Similarity를 이용한 문서 검색 방식으로 구현하였다. 검색된 문서를 모델에 전달하는 신경망 생성 방식과 문서를 직접 반환하는 근거 기반 방식을 비교하였다.

### 8.4. Dialogue 모델 추가 학습 결과

| 평가 항목 | 추가 학습 전 MHA | Dialogue |
|---|---:|---:|
| 단기 기억 | 0% | 25% |
| 기억 정보 변경 | 0% | 20% |
| 장기 기억 문맥 활용 | 0% | 15% |
| Persona 템플릿 | 0% | 100% |
| 문서 기반 QA | 0% | 7.5% |

위 결과는 합성 평가 문맥을 제공한 통제 실험에서 측정하였다.

SQLite 기반 기억 저장 및 문서 직접 반환 기능은 별도의 10개 기능 검사에서 모두 성공하였다. 그러나 실제 생성 답변을 다음 대화에 다시 사용하는 자유 대화에서는 기억 활용 능력이 충분하지 않았다.

외부 정보를 정확하게 저장하고 검색하는 기능과 언어 모델이 해당 정보를 이해하여 답변하는 능력은 서로 다른 문제임을 확인하였다.

---

## 9. GRPO 기반 강화학습 실험

DeepSeek-R1에서 사용된 GRPO(Group Relative Policy Optimization)의 핵심 원리를 확인하기 위해 한 자리 수 덧셈 문제를 대상으로 축소 실험을 수행하였다.

정답이 명확한 수학 문제를 이용해 보상 함수를 구성하고, 동일한 질문에서 생성된 여러 답변의 상대적인 보상을 기반으로 Advantage를 계산하였다.

기존 지도학습 모델과 추가 지도학습 대조군, GRPO 모델의 성능을 비교하였다.

| Model | Test Pass@1 |
|---|---:|
| Math-SFT | 0% |
| Math-SFT-control | 0% |
| Math-GRPO | 0% |

일부 응답 그룹에서 보상 차이와 정책 학습 신호가 발생하였으나, 최종 테스트에서는 GRPO에 의한 정답률 향상을 확인하지 못하였다.

기본 모델의 문제 해결 능력과 데이터의 규모, 보상 설계가 충분하지 않은 상태에서는 강화학습만으로 성능을 개선하기 어렵다는 것을 확인하였다. 테스트 문제도 8개로 제한되어 있으므로 일반적인 GRPO의 성능에 대한 결론으로 확대할 수는 없다.

---

## 10. 사전학습 Qwen 및 LoRA 비교

직접 구현한 소형 Transformer와 대규모 사전학습 모델의 차이를 확인하기 위해 Hugging Face에서 공개한 Qwen2.5 모델을 사용하였다.

Qwen2.5-0.5B-Instruct에는 한국어 챗봇 데이터를 이용하여 LoRA 기반 추가 지도학습을 수행하였다.

| Model | Parameter Scale | Character LCS F1 |
|---|---:|---:|
| Custom MHA | 1.30M | 0.2572 |
| Qwen2.5-0.5B | 0.5B | 0.0914 |
| Qwen2.5-0.5B + LoRA | 0.5B | 0.2473 |
| Qwen2.5-1.5B | 1.5B | 0.0910 |

Qwen 0.5B에 LoRA 추가 학습을 적용한 이후 한국어 챗봇 데이터의 답변 표현과 유사한 문장을 생성하는 경향이 증가하였다.

반면 사전학습 모델은 기존 데이터의 짧은 정답 문장과 다른 표현이나 상대적으로 긴 답변을 생성하기도 하였다. 따라서 문자 유사도 결과만으로 자체 구현 모델과 사전학습 언어 모델의 실제 대화 능력을 비교하기는 어렵다.

별도의 기억 실험에서는 Qwen 1.5B와 LoRA 적용 모델이 앞선 대화에서 제공한 로봇 이름을 올바르게 반환하였다.

---

## 11. Gradio 기반 챗봇 인터페이스

학습된 모델을 직접 테스트할 수 있도록 Gradio 기반의 대화 인터페이스를 구성하였다.

| 기능 | 구현 내용 |
|---|---|
| Model Selection | 자체 구현 Transformer 및 Qwen 계열 모델 선택 |
| System Prompt | 챗봇 역할 및 상황 설정 |
| Short-term Memory | 이전 대화 문맥 활용 |
| Long-term Memory | SQLite 기반 정보 저장 및 검색 |
| RAG | 관련 문서 검색 및 문맥 전달 |
| Decoding | Greedy, Beam, Top-k, Top-p 선택 |
| Temperature | Sampling 확률 분포 조절 |
| Tool | 명시적인 기억 관리 및 계산기 호출 |
| Experiment Recording | 대화 내용, 설정값, 응답 경로 및 실험 결과 저장 |
| Runtime Recovery | 체크포인트 및 토크나이저 복원 기능 |

인터페이스를 통해 모델별 답변 비교, 다중 턴 대화, 기억 정보 변경, 역할 설정 및 검색 기반 답변 등을 실험할 수 있도록 구성하였다.

다만 마지막 실행에서는 Colab 런타임 초기화로 자체 모델과 LoRA 가중치를 복원하지 못하였다. 따라서 인터페이스 구성 및 실행과 모든 학습 모델의 실제 대화 연결 검증을 구분하였다.

---

## 12. 실행 방법

1. 프로젝트 Notebook을 Google Colab에서 실행한다.
2. 런타임 유형을 GPU로 설정한다.
3. Step 0에서 실행 모드와 추가 실험 옵션을 설정한다.
4. Step 1부터 Step 5까지 순차적으로 실행하여 기본 한국어 Transformer를 학습하고 평가한다.
5. Step 6 이후의 Attention 비교, 기억 시스템, GRPO 및 Qwen 실험은 필요한 범위에서 추가로 실행한다.
6. 학습 결과와 모델 체크포인트를 별도로 저장한다.
7. 챗봇 인터페이스를 사용하려는 경우 Step 16의 Backend를 초기화한 뒤 RUN_UI를 True로 설정한다.

기본 과제의 구현 및 평가는 Step 1~5에서 완료되며, 이후의 실험은 Transformer 구조와 대화 시스템에 대한 이해를 확장하기 위한 심화 과정이다.

Qwen 및 LoRA 실험은 Hugging Face 모델 다운로드가 필요하다. Colab의 PyTorch, Transformers 및 PEFT 버전에 따라 라이브러리 호환성 문제가 발생할 수 있으므로 실행 환경을 확인해야 한다.

Notebook에는 기존 GPU 학습에서 얻은 실제 실행 결과가 포함되어 있으며, 재실행 환경에 따라 측정값은 달라질 수 있다.

---

## 13. 프로젝트 평가 및 회고

이번 프로젝트를 통해 Transformer의 기본 구조뿐 아니라 Attention 연산 방식, 토큰 생성 과정, KV Cache, Latent Compression 및 Sparse Attention이 실제 코드에서 어떻게 구현되는지 확인할 수 있었다.

처음에는 Attention 구조를 개선하고 기억 기능을 추가하면 챗봇의 대화 능력도 함께 향상될 것으로 예상하였다. 그러나 실험을 진행하면서 연산 효율, 언어 모델의 문맥 이해 능력, 외부 정보 저장 및 검색 기능은 서로 구분하여 접근해야 하는 문제라는 것을 알게 되었다.

실제로 Attention 구조에 따라 Cache 저장량을 상당히 줄일 수 있었지만 추론 속도가 반드시 향상되는 것은 아니었다. 또한 학습 손실이 감소하더라도 질문과 관계없는 답변을 생성하거나 이전 대화를 반복하는 문제가 발생하였다.

이러한 결과를 통해 모델 구조 자체의 개선만큼 학습 데이터의 규모와 다양성, 학습 목적, 평가 방법이 중요하다는 점을 확인하였다. 특히 단일 문답 중심의 데이터만으로는 자연스러운 다중 턴 대화와 새로운 상황에 대한 일반화 능력을 확보하기 어렵다는 한계가 있었다.

향후에는 실제 한국어 다중 턴 대화 데이터를 활용하여 문맥 이해 능력을 개선하고, 충분한 사전학습이 이루어진 모델을 기반으로 추가 학습을 진행하고자 한다. 또한 Loss와 문자 유사도뿐 아니라 질문 적합성, 문맥 일관성 및 기억 정확도를 함께 고려하는 평가 방법을 적용할 필요가 있다고 생각한다.

모든 추가 실험에서 성능 향상을 달성하지는 못했지만, 이론적으로 학습한 기술을 직접 구현하고 정량적인 결과를 바탕으로 실제 효과와 한계를 구분할 수 있었다는 점에서 의미 있는 프로젝트였다.

---

## 14. References

| 자료 | Reference |
|---|---|
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| SentencePiece | [SentencePiece Official Repository](https://github.com/google/sentencepiece) |
| Korean Chatbot Dataset | [songys/Chatbot_data](https://github.com/songys/Chatbot_data) |
| MQA | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| GQA | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) |
| MLA | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| DSA | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| GRPO | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| LoRA | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) |
| Qwen | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Qwen | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
