# 4주차 논문 정리: CLIP & SAM

- **Part 1. CLIP**: 발제 27기 김지우, 이서진
- **Part 2. SAM**: 발제 27기 한정재, 이수빈
- CLIP vs SAM 비교표는 SAM 파트 9장에 있음

---

## CLIP: Learning Transferable Visual Models From Natural Language Supervision

- **저자**: Alec Radford, Jong Wook Kim 외 (OpenAI), ICML 2021
- **논문**: https://arxiv.org/pdf/2103.00020
- **발제**: 27기 김지우, 이서진

---

### 0. 한 줄 요약

> 인터넷에서 모은 **4억 개의 (이미지, 텍스트) 쌍**으로 "이 이미지에 어떤 텍스트가 짝인가"를 맞히는 **대조 학습(contrastive learning)** 을 했더니, 라벨 없이도 텍스트만 바꿔 넣으면 어떤 분류 문제든 풀 수 있는 **zero-shot** 비전 모델이 나왔다.

---

### 1. 배경 & 문제의식

#### 기존 비전 모델의 한계
- ImageNet처럼 **고정된 클래스 집합(예: 1000개)** 에 대해 사람이 라벨링한 데이터로 지도학습
- 새로운 개념을 인식하려면 → 추가 라벨 데이터 + 재학습(fine-tuning) 필요
- 라벨링 비용이 크고, 학습한 클래스 밖으로 일반화가 안 됨

#### NLP에서 얻은 힌트
- GPT-3 등은 웹의 raw text로 사전학습 → task-agnostic하게 zero-shot 전이 가능
- "웹 스케일 텍스트 = 라벨보다 훨씬 풍부한 supervision"
- 그렇다면 **이미지도 자연어로 supervision** 하면 되지 않을까?

#### 기존 시도들 (VirTex, ICMLM, ConVIRT 등)
- 자연어로 이미지 표현을 학습하는 아이디어 자체는 있었음
- 하지만 데이터 규모가 작아서(수십만~백만 단위) 성능이 기존 지도학습에 크게 못 미침
- CLIP의 핵심 기여 = **규모(scale)** + **효율적인 학습 목표(contrastive)**

---

### 2. 데이터셋: WIT (WebImageText)

- 인터넷에서 수집한 **4억(400M) 개의 (image, text) 쌍**
- 50만 개의 쿼리(위키피디아 빈출 단어 등)로 검색해 수집, 쿼리당 최대 2만 쌍으로 클래스 균형 조정
- 총 단어 수는 GPT-2 학습에 쓴 WebText와 비슷한 수준

---

### 3. 방법론

#### 3.1 왜 Contrastive인가? — 학습 효율 비교

| 방식 | 설명 | 효율 |
|---|---|---|
| Transformer LM | 이미지 보고 캡션을 **단어 단위로 정확히 생성** | 가장 느림 |
| Bag-of-Words 예측 | 캡션의 단어 집합을 예측 | LM보다 3배 빠름 |
| **Contrastive (CLIP)** | 어떤 텍스트가 **통째로** 이 이미지와 짝인지만 맞힘 | BoW보다 추가로 **4배** 빠름 |

- 캡션의 정확한 단어를 맞히는 건 너무 어려운 문제 (같은 이미지에도 설명이 다양함)
- "짝 맞추기"로 문제를 쉽게 만들어서 학습 효율을 크게 올림

#### 3.2 구조

```
이미지 → Image Encoder (ResNet or ViT) → 선형 투영 → L2 정규화 → I_e (N × d)
텍스트 → Text Encoder (Transformer)     → 선형 투영 → L2 정규화 → T_e (N × d)

logits = (I_e · T_eᵀ) × exp(t)      # N × N 코사인 유사도 행렬, t = 학습 가능한 temperature
loss   = (CE(logits, 행방향) + CE(logits, 열방향)) / 2   # 대칭 cross-entropy
```

- 배치에 N개의 (이미지, 텍스트) 쌍이 있으면 N×N 유사도 행렬을 만듦
- **대각선(N개) = 정답 쌍 → 유사도 ↑**, 나머지 N²−N개 = 오답 쌍 → 유사도 ↓
- 이미지→텍스트, 텍스트→이미지 양방향 cross-entropy의 평균 (InfoNCE loss)

##### 논문의 Numpy-like 의사코드
```python
I_f = image_encoder(I)          # [n, d_i]
T_f = text_encoder(T)           # [n, d_t]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)   # [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)   # [n, d_e]
logits = np.dot(I_e, T_e.T) * np.exp(t)        # [n, n]
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

#### 3.3 인코더 상세
- **Image Encoder**: 두 계열 실험
  - ResNet 계열 (RN50, RN101, RN50x4, RN50x16, RN50x64) — attention pooling 등 수정
  - Vision Transformer (ViT-B/32, ViT-B/16, ViT-L/14) — 약간의 layer norm 추가
- **Text Encoder**: Transformer (63M 파라미터, 12층, width 512, 8 heads)
  - BPE 토크나이저(vocab 49,152), 최대 길이 76 토큰
  - `[EOS]` 토큰 위치의 출력을 텍스트 표현으로 사용
- 두 인코더 모두 **처음부터(from scratch) 학습** (ImageNet 사전학습 가중치 사용 X)

#### 3.4 학습 디테일
- 32 epoch, Adam, cosine schedule
- **배치 크기 32,768** (대조 학습은 negative가 많을수록 유리)
- temperature τ는 학습 가능한 파라미터 (0.07로 초기화, 로짓 스케일 최대 100으로 클리핑)
- 데이터 증강은 random square crop 정도만 사용
- 가장 큰 모델: RN50x64는 V100 592장으로 18일, ViT-L/14는 V100 256장으로 12일
- 최종 대표 모델: **ViT-L/14@336px** (336 해상도로 1 epoch 추가 학습)

---

### 4. Zero-shot 분류는 어떻게 하나?

1. 분류할 클래스 이름들을 텍스트로 만든다 → `"A photo of a {dog}."`, `"A photo of a {cat}."` ...
2. 모든 텍스트를 Text Encoder에 넣어 임베딩 → 이것이 곧 **분류기 가중치** 역할
3. 이미지를 Image Encoder에 넣어 임베딩
4. 이미지 임베딩과 각 텍스트 임베딩의 코사인 유사도 → softmax → 가장 높은 클래스 선택

> 텍스트 인코더가 **"hypernetwork"처럼 분류기의 가중치를 생성**해준다고 볼 수 있음. 클래스를 바꾸고 싶으면 텍스트만 바꾸면 됨.

#### Prompt Engineering & Ensembling
- 라벨 단어 하나만 넣으면(`"dog"`) 성능이 떨어짐 → 학습 데이터의 텍스트는 대부분 **문장** 형태이기 때문
- 다의어 문제: "crane"(두루미/크레인), "boxer"(개 품종/권투선수)
- `"A photo of a {label}."` 템플릿만으로 ImageNet 정확도 **+1.3%p**
- 데이터셋별 맥락 추가: `"A photo of a {label}, a type of pet."`, `"a satellite photo of a {label}."`
- 여러 템플릿(ImageNet은 80개) 임베딩을 평균내는 **ensembling** → 추가 향상
- prompt engineering + ensembling 합쳐서 ImageNet 약 **+5%p**

---

### 5. 실험 결과

#### 5.1 Zero-shot 성능
- **ImageNet zero-shot 76.2%** → ImageNet 128만 장으로 **지도학습한 ResNet-50(76.1%)과 동급**, 라벨 이미지를 하나도 안 쓰고!
- 기존 zero-shot 방법(Visual N-Grams, 11.5%) 대비 압도적 향상
- 27개 데이터셋 중 **16개에서 ResNet-50 linear probe(지도학습 베이스라인)를 zero-shot으로 이김**
  - 강한 곳: 일반 객체 분류, 행동 인식(Kinetics700, UCF101), STL10
  - 약한 곳: 위성사진(EuroSAT), 림프절 종양(PatchCamelyon), 객체 개수 세기(CLEVRCounts), 교통 표지판(GTSRB) 등 **전문적/추상적 task**

#### 5.2 Few-shot과 비교
- CLIP **zero-shot ≈ CLIP 4-shot linear probe** 성능
- 다른 모델들의 16-shot linear probe와 비슷한 수준
- 흥미로운 점: 사람은 1장만 보여줘도 성능이 크게 오르는데, CLIP은 zero-shot → 1-shot에서 오히려 떨어지기도 함 (사전지식 활용 방식의 차이)

#### 5.3 Representation Learning (Linear Probe)
- 이미지 인코더 고정 + 로지스틱 회귀만 학습
- 가장 큰 CLIP 모델이 기존 최고 모델(Noisy Student EfficientNet-L2 등)보다 27개 데이터셋 평균에서 우수
- 연산량 대비 효율도 좋음, 특히 ViT 기반 CLIP이 ResNet 기반보다 약 3배 연산 효율적

#### 5.4 Distribution Shift에 대한 Robustness ⭐
- ImageNet으로 학습한 모델은 ImageNet에선 잘해도 **ImageNetV2, ImageNet-R(렌더링/만화), ImageNet Sketch, ObjectNet, ImageNet-A(적대적)** 에서 성능 급락
- Zero-shot CLIP은 ImageNet 정확도가 같은 모델 대비 분포 변화 데이터셋에서 훨씬 덜 떨어짐 → **robustness gap을 최대 75%까지 줄임**
  - 예: ImageNet-R — ResNet-101 37.7% vs **CLIP 88.9%**
  - ImageNet Sketch — 25.2% vs **60.2%**
- 반대로 CLIP을 ImageNet에 맞춰 적응(logistic regression)시키면 ImageNet 정확도는 +9.2%p 오르지만 다른 분포 데이터셋 평균 robustness는 오히려 약간 떨어짐
  → 특정 데이터셋에 맞출수록 그 분포의 spurious correlation을 학습한다는 시사점

#### 5.5 사람과의 비교
- Oxford-IIIT Pets로 사람 실험: 사람 zero-shot 53.7% → 1-shot 75.7%로 급상승
- CLIP zero-shot은 93.5%로 사람보다 높지만, 사람처럼 **적은 예시에서 효율적으로 배우는 능력**은 부족

---

### 6. 데이터 중복(Overlap) 분석
- 4억 장 인터넷 데이터에 평가 데이터셋 이미지가 포함되었을 가능성 점검
- 중복 탐지기로 검사 → 35개 중 9개 데이터셋은 중복 없음, 중앙값 중복률 2.2%
- 전체 성능 영향은 대부분 **0.1% 이내**로 무시할 수준 (최대 Birdsnap 0.6%p)

---

### 7. 한계점 (Limitations)
1. **여전히 SOTA와 차이 큼**: zero-shot CLIP은 ResNet-50 베이스라인 수준일 뿐, 각 task의 SOTA와는 격차 있음. 논문 추정으로는 SOTA 도달에 약 1000배 연산 필요
2. **약한 task**: 세부 분류(차종, 꽃 종, 비행기 기종), 추상적/체계적 task(개수 세기), 새로운 task(사진 속 가장 가까운 차까지의 거리)
3. **진짜 out-of-distribution엔 약함**: MNIST 손글씨 정확도 88% → 단순 로지스틱 회귀(raw pixel)보다도 낮음. 인터넷 데이터에 손글씨 숫자가 거의 없었기 때문
4. **생성 불가**: 주어진 후보 텍스트 중에서 고를 뿐, 캡션을 직접 생성하지는 못함
5. **데이터 효율성 낮음**: 4억 장 × 32 epoch = 128억 장을 봄
6. **평가 방식 문제**: zero-shot이라 하면서 validation set을 보며 프롬프트/모델을 고름
7. **사회적 편향**: 필터링되지 않은 인터넷 데이터 → 사회적 편향 학습

---

### 8. 사회적 영향 (Broader Impacts)
- **편향**: FairFace 실험에서 흑인 이미지가 "비인간(non-human)" 범주로 오분류되는 비율이 높게 나타남, 범죄 관련 클래스에 특정 인종/연령이 더 많이 분류됨
  - 클래스 설계(어떤 라벨 후보를 넣는지)에 따라 편향 양상이 크게 달라짐 → zero-shot 모델은 **클래스 설계 자체가 성능과 편향에 영향**
- **감시(surveillance)**: CCTV 이미지 분류, 유명인 식별(zero-shot 100 클래스 기준 59.2%) 등 악용 가능성 논의

---

### 9. 의의 & 후속 영향
- 비전과 언어를 **하나의 공유 임베딩 공간(joint embedding space)** 에 정렬 → 멀티모달 AI의 기반
- 후속 활용
  - **텍스트→이미지 생성**: DALL·E 2(unCLIP), Stable Diffusion(CLIP 텍스트 인코더 사용)
  - **Open-vocabulary 검출/분할**: ViLD, OWL-ViT, LSeg, GroupViT 등
  - **이미지-텍스트 검색**, 멀티모달 LLM(LLaVA 등)의 비전 인코더
  - **SAM의 text prompt 실험에서도 CLIP 텍스트 인코더 사용** (→ SAM 정리 참고)
- 후속 연구: ALIGN(Google, 18억 쌍 noisy data), OpenCLIP/LAION, SigLIP(softmax 대신 sigmoid loss) 등

---

### 10. 핵심 키워드 정리

| 키워드 | 의미 |
|---|---|
| Natural Language Supervision | 라벨 대신 자연어 텍스트를 supervision으로 사용 |
| Contrastive Learning | 짝이 맞는 쌍은 가깝게, 틀린 쌍은 멀게 |
| InfoNCE / 대칭 CE Loss | N×N 유사도 행렬의 행·열 방향 cross-entropy 평균 |
| Zero-shot Transfer | 추가 학습 없이 텍스트 프롬프트만으로 새 task 수행 |
| Prompt Engineering / Ensembling | "A photo of a {label}." 템플릿 + 여러 템플릿 평균 |
| Linear Probe | 인코더 고정, 선형 분류기만 학습해 표현 품질 평가 |
| Effective Robustness | 분포 변화 시 성능 하락이 적은 정도 |

---

### 💬 생각해볼 질문
- 왜 배치 크기가 클수록 contrastive learning에 유리할까? (negative sample 수)
- zero-shot 분류에서 클래스 후보 문장 설계가 편향에 어떤 영향을 줄 수 있을까?
- CLIP이 MNIST에 약한 이유와, "데이터 분포가 넓으면 일반화된다"는 가정의 한계는?
- 생성형 목표(캡션 생성) 대신 대조 목표를 택한 trade-off는 무엇인가?

---

## SAM: Segment Anything

- **저자**: Alexander Kirillov, Eric Mintun, Nikhila Ravi 외 (Meta AI Research, FAIR), ICCV 2023
- **논문**: https://arxiv.org/pdf/2304.02643
- **발제**: 27기 한정재, 이수빈

---

### 0. 한 줄 요약

> NLP의 foundation model처럼 **이미지 분할(segmentation)용 foundation model**을 만들기 위해, ① **Promptable Segmentation**이라는 새 task, ② 이를 푸는 모델 **SAM**, ③ 모델과 함께 데이터를 만드는 **Data Engine**으로 만든 **11억 개 마스크 데이터셋 SA-1B**를 제시했다.

---

### 1. 배경 & 문제의식

- NLP: GPT 같은 대규모 모델 + **프롬프트**로 처음 보는 task도 zero-shot 수행
- 비전: CLIP이 이미지-텍스트 정렬로 비슷한 가능성을 보여줌
- 하지만 **segmentation**은?
  - 기존 모델은 특정 데이터셋/클래스에 맞춰 학습 (COCO 80개 클래스 등)
  - 픽셀 단위 마스크 라벨링은 매우 비쌈 → 웹에 대규모 마스크 데이터가 존재하지 않음
- 논문의 질문: **"segmentation의 foundation model을 만들려면 무엇이 필요한가?"**
  1. **Task**: zero-shot 일반화를 가능하게 하는 task는?
  2. **Model**: 그 task를 풀 모델 구조는?
  3. **Data**: 학습에 필요한 대규모 데이터는 어떻게 구하나?

---

### 2. Task: Promptable Segmentation

- **프롬프트**가 주어지면 그에 해당하는 **유효한(valid) 마스크**를 출력하는 task
- 프롬프트 종류
  - **Point** (전경/배경 점 클릭)
  - **Box** (바운딩 박스)
  - **Mask** (대략적인 마스크)
  - **Text** (자유 텍스트 — 논문에선 탐색적 실험 수준)

#### 모호성(Ambiguity) 처리 ⭐
- 셔츠 위에 점 하나를 찍으면 → "셔츠"인지 "사람 전체"인지 모호
- 요구사항: 프롬프트가 모호해도 **그중 적어도 하나의 합리적인 마스크**를 출력해야 함
  (LLM이 모호한 질문에도 일관된 답을 내는 것과 유사)

#### 왜 이 task인가?
- 자연스러운 **사전학습 목표**가 됨 (다양한 프롬프트 시뮬레이션으로 학습)
- **prompt engineering으로 downstream task에 전이** 가능
  - 예: 객체 검출기의 box 출력 → SAM의 box prompt → **instance segmentation**
- 여러 시스템의 **구성 요소(composable component)** 로 쓰일 수 있음

---

### 3. Model: Segment Anything Model (SAM)

```
         ┌────────────────────┐
이미지 →  │  Image Encoder     │ → image embedding (64×64×256)  ─┐
         │  (MAE ViT-H/16)    │     ※ 이미지당 1번만 계산         │
         └────────────────────┘                                 ▼
                                                      ┌──────────────────┐
프롬프트 → Prompt Encoder → prompt tokens ───────────→ │  Mask Decoder    │ → 마스크 3개 + IoU 점수
(점/박스/텍스트: sparse, 마스크: dense)                 │  (경량, ~50ms)   │
                                                      └──────────────────┘
```

#### 3.1 Image Encoder (무거움, 1회 실행)
- **MAE(Masked Autoencoder)로 사전학습된 ViT-H/16** (고해상도 입력을 처리하도록 최소한만 수정)
- 입력 1024×1024 → 출력 **64×64 image embedding** (16배 다운스케일), 채널 256으로 축소
- 이미지 한 장에 **한 번만** 계산 → 이후 프롬프트를 여러 번 바꿔도 재사용 (amortize)

#### 3.2 Prompt Encoder
- **Sparse prompt** (점, 박스, 텍스트)
  - 점/박스: **positional encoding** + 프롬프트 종류별 학습된 embedding의 합
    - 점: 위치 인코딩 + 전경/배경 embedding
    - 박스: 좌상단/우하단 두 점으로 표현
  - 텍스트: **CLIP의 텍스트 인코더** 사용
- **Dense prompt** (마스크)
  - 합성곱(conv)으로 다운샘플해 image embedding과 **element-wise로 더함**

#### 3.3 Mask Decoder (가벼움, 빠름)
- 수정된 **Transformer decoder 블록 2개**
- 각 블록에서 4단계:
  1. 토큰(프롬프트 + output token)끼리 **self-attention**
  2. 토큰 → 이미지 **cross-attention** (token-to-image)
  3. 토큰별 **MLP**
  4. 이미지 → 토큰 **cross-attention** (image-to-token) → 이미지 임베딩도 프롬프트 정보로 업데이트
- 이후 image embedding을 transposed conv로 **4배 업샘플**
- output token을 MLP에 통과시킨 벡터와 업샘플된 임베딩을 **내적 → 픽셀별 마스크 확률**
  (dynamic linear classifier)

#### 3.4 모호성 대응: 다중 마스크 출력
- 하나의 프롬프트에 대해 **마스크 3개** 출력 (대체로 whole / part / subpart 계층)
- 학습 시 3개 중 **loss가 가장 작은 것에만** 역전파 (minimum loss)
- 각 마스크의 품질을 추정하는 **IoU prediction head** 추가 → 순위 매기기용
- 프롬프트가 여러 개(모호하지 않음)면 별도의 4번째 output token으로 마스크 1개만 출력

#### 3.5 효율성
- 이미지 임베딩이 미리 계산되어 있으면 prompt encoder + mask decoder는 **웹 브라우저 CPU에서 ~50ms**
- → **실시간 인터랙티브** 분할 가능

#### 3.6 학습
- **Loss**: Focal loss : Dice loss = **20 : 1** 선형 결합 (마스크) + IoU 예측엔 MSE
- **인터랙티브 설정 시뮬레이션**: 한 마스크당 **11 라운드**
  - 첫 프롬프트는 점 또는 박스(GT에서 노이즈 추가)
  - 이후 이전 예측과 GT의 **오류 영역에서 다음 점을 샘플링** (FN 영역이면 전경점, FP 영역이면 배경점)
  - 이전 예측 마스크(logits)도 다음 라운드 프롬프트로 입력
  - 중간에 새 점 없이 이전 마스크만 넣는 라운드도 포함 → 마스크 자체 개선 학습

---

### 4. Data: Data Engine → SA-1B

웹에 대규모 마스크 데이터가 없으므로 **모델이 데이터를 만들고, 그 데이터로 모델을 개선하는 루프(model-in-the-loop)** 를 설계.

| 단계 | 방식 | 규모 | 특징 |
|---|---|---|---|
| **① Assisted-manual** | 전문 어노테이터가 SAM 기반 브라우저 툴로 점 클릭 + 브러시/지우개 수정 | 12만 장, **430만 마스크** | 라벨 이름 제약 없음, 두드러진 객체부터. 마스크당 34초 → 14초로 단축(COCO보다 6.5배 빠름). 모델 6번 재학습 |
| **② Semi-automatic** | SAM이 확신 있는 마스크를 먼저 자동 채우고, 사람은 **놓친 객체**만 추가 | 18만 장, **+590만 마스크** (누적 1,020만) | **다양성** 향상. 이미지당 마스크 44 → 72개. 5번 재학습 |
| **③ Fully automatic** | 사람 없이 **32×32 격자 점**으로 프롬프트 → 자동 생성 | **1,100만 장, 11억 마스크** | 모호성 인식 모델 덕분에 가능 |

#### Fully automatic 단계의 품질 필터링
- **IoU 예측 점수**가 높은 마스크만 (confident)
- **Stability**: 확률 임계값을 0.5−δ, 0.5+δ로 바꿔도 마스크가 거의 같으면 안정적 → 채택
- **NMS**로 중복 제거
- 작은 마스크 품질 개선을 위해 확대(zoom-in)된 crop에서도 추가 처리

#### SA-1B 데이터셋
- **1,100만 장 이미지, 11억(1.1B) 마스크** → 기존 최대 분할 데이터셋(Open Images) 대비 **이미지 11배, 마스크 400배**
- 이미지: 라이선스 받은 고해상도 사진(평균 3300×4950) → 짧은 변 1500px로 다운샘플해 공개
- **얼굴과 차량 번호판은 블러 처리** (프라이버시)
- 마스크의 **99.1%가 자동 생성**
- 품질 검증: 샘플 500장(약 5만 마스크)을 전문가가 수정 → **94%의 쌍이 IoU > 90%**, 97%가 IoU > 75%
  (사람 간 일치도가 85~91% IoU 수준임을 고려하면 매우 높음)
- 특징: 기존 데이터셋보다 **이미지 전체에 고르게 퍼진 마스크**, 작은/중간 크기 객체 비율 높음, 이미지당 마스크 수 많음(평균 약 100개)
- **Responsible AI**: 지역·소득 수준별 이미지 분포 분석(유럽/아시아·오세아니아 비중 큼, 아프리카·저소득 국가도 기존 데이터셋보다 높은 비율), 성별·피부색·연령대별 사람 분할 성능 차이가 크지 않음을 확인

---

### 5. 실험: Zero-shot Transfer

SAM은 SA-1B로만 학습하고, **본 적 없는 데이터셋/task**에 프롬프트 설계만으로 적용.

#### 5.1 Single Point → Valid Mask (핵심 실험)
- 새로 구성한 **23개 다양한 데이터셋** (수중, 자아 시점, 의료/X-ray, 그림, 드라이빙 등)
- 비교 대상: RITM (당시 강력한 인터랙티브 분할 모델)
- 자동 지표(mIoU): 23개 중 **16개에서 SAM이 우수**, "oracle"(3개 중 GT와 가장 맞는 것 선택) 기준이면 전 데이터셋에서 우수
- **사람 평가(1~10점 품질 점수)**: SAM이 RITM보다 일관되게 높음 (7~9점대)
  → 자동 지표는 GT가 한 가지 해석만 정답이라 모호성 상황에서 SAM을 과소평가하는 경향
- 점을 여러 개 주면 SAM과 기존 방법의 차이는 줄어듦 (SAM의 강점은 **적은 프롬프트, 특히 1개의 점**일 때)

#### 5.2 Edge Detection (BSDS500)
- 16×16 격자 점 → 자동 마스크 생성 → Sobel 필터로 마스크 경계 추출 → edge map
- **엣지 검출로 학습한 적이 없는데도** 합리적인 엣지 맵 생성 (정답에 없는 엣지도 잡아서 precision은 낮지만 recall 높음)

#### 5.3 Object Proposals (LVIS)
- 자동 마스크 생성 결과를 object proposal로 사용
- 강한 베이스라인 ViTDet-H 대비 전체 AR@1000은 약간 낮지만, **중간/큰 객체, 희귀(rare)·일반(common) 객체에서는 더 우수**

#### 5.4 Instance Segmentation (COCO, LVIS)
- **ViTDet 검출기의 박스를 box prompt로 SAM에 입력**
- mask AP는 ViTDet보다 약간 낮음 (COCO 46.5 vs 51.0 수준)
- 하지만 **사람 평가에선 SAM의 마스크 품질이 더 높게 평가**됨 → 경계가 더 깔끔. COCO/LVIS GT 자체의 노이즈 편향을 SAM은 학습하지 않았기 때문

#### 5.5 Text → Mask (탐색적, proof-of-concept)
- 학습 시: 마스크 영역 이미지의 **CLIP 이미지 임베딩**을 프롬프트로 사용 (텍스트 라벨 없이)
- 추론 시: CLIP은 이미지-텍스트 임베딩이 정렬되어 있으므로 **CLIP 텍스트 임베딩**을 대신 넣음
- "a wheel", "beaver tooth grille" 같은 텍스트로 분할 가능. 실패 시 점을 추가하면 보정됨
- → **CLIP의 공유 임베딩 공간 덕분에 가능한 트릭** (CLIP 논문과 연결되는 지점!)

#### 5.6 Ablation
- 데이터 엔진 단계별 데이터 추가 → 성능 향상, 자동 마스크만으로 학습해도 전체 데이터 사용 대비 성능 차이 미미(≈0.5 mIoU)
- **데이터 양**: 1,100만 장 대신 **100만 장(약 1억 마스크)만 써도 거의 비슷한 성능**
- **인코더 크기**: ViT-B → ViT-L 향상 크지만, ViT-L → ViT-H는 향상 폭 작음

---

### 6. 한계점 (Limitations)
1. **세밀한 구조**를 놓치거나, 작은 떨어진 조각을 잘못 만들어내기도 함 (hallucinate)
2. 경계가 zoom-in 기반 고해상도 전문 방법만큼 **선명(crisp)하지 않음**
3. 점을 **많이 줄 때**는 전용 인터랙티브 분할 방법보다 못할 수 있음 (SAM은 범용성 우선)
4. 이미지 인코더(ViT-H)가 무거워 **전체 파이프라인은 실시간이 아님** (디코더만 실시간)
5. **Text-to-mask는 초기 단계**, 견고하지 않음
6. **Semantic/panoptic segmentation**을 프롬프트만으로 수행하는 방법은 불명확
7. 특정 도메인(의료 등)에선 전문 모델보다 성능 낮을 수 있음

---

### 7. 의의 & 후속 영향
- **비전 분할 분야의 foundation model 패러다임** 제시 (task + model + data를 함께 설계)
- **Data Engine**: 모델-데이터 공진화(model-in-the-loop) 방법론의 대표 사례
- 모델 가중치와 SA-1B 데이터셋 **오픈 공개**
- 후속 연구
  - **경량화**: MobileSAM, EfficientSAM, FastSAM, EfficientViT-SAM
  - **도메인 특화**: MedSAM (의료 영상)
  - **텍스트 결합**: Grounded-SAM (Grounding DINO로 텍스트→박스 → SAM으로 마스크)
  - **확장**: HQ-SAM(고품질 경계), Semantic-SAM, **SAM 2**(비디오까지 확장, 메모리 모듈)
  - 3D, 트래킹, 인페인팅(Inpaint Anything) 등의 구성 요소로 활용

---

### 8. 핵심 키워드 정리

| 키워드 | 의미 |
|---|---|
| Promptable Segmentation | 점/박스/마스크/텍스트 프롬프트 → 유효한 마스크 |
| Valid Mask | 모호한 프롬프트라도 가능한 해석 중 하나에 맞는 합리적 마스크 |
| Ambiguity-aware | 마스크 3개(whole/part/subpart) + IoU 점수 출력 |
| Image Encoder (MAE ViT-H) | 무겁지만 이미지당 1회, 임베딩 재사용 |
| Prompt Encoder | sparse(위치 인코딩, CLIP 텍스트) / dense(conv) |
| Mask Decoder | 양방향 cross-attention 2블록, ~50ms |
| Data Engine | assisted-manual → semi-automatic → fully automatic |
| SA-1B | 1,100만 이미지, 11억 마스크 |
| Zero-shot Transfer | 프롬프트 설계로 엣지 검출, proposal, instance seg 등에 전이 |

---

### 9. CLIP vs SAM 비교

| | CLIP | SAM |
|---|---|---|
| 목적 | 이미지 **이해/분류** (무엇인가?) | 이미지 **분할** (어디인가?) |
| 출력 | 이미지·텍스트 임베딩 → 유사도 | 픽셀 단위 마스크 |
| 프롬프트 | 텍스트 ("A photo of a {label}") | 점, 박스, 마스크, (텍스트) |
| 데이터 | 웹 크롤링 4억 이미지-텍스트 쌍 (이미 존재) | Data Engine으로 **직접 생성**한 11억 마스크 |
| 학습 목표 | 대조 학습 (InfoNCE) | 인터랙티브 시뮬레이션 + Focal/Dice loss |
| 의미(semantic) 인식 | O (클래스 개념 이해) | X (마스크는 있지만 "무엇인지"는 모름) |
| 연결점 | SAM 텍스트 프롬프트에 CLIP 텍스트 인코더 사용 | Grounded-SAM 등에서 CLIP 계열과 조합 |

> 공통점: 둘 다 **"대규모 데이터 + 프롬프트 기반 zero-shot 전이"** 라는 NLP foundation model 패러다임을 비전으로 가져온 연구

---

### 💬 생각해볼 질문
- 왜 이미지 인코더는 무겁게, 마스크 디코더는 가볍게 설계했을까? (인터랙티브 사용 시나리오)
- 모호성을 다루기 위해 마스크 3개 + minimum loss를 쓴 이유는? 1개만 출력하면 어떤 문제가? (여러 정답의 평균 → 흐릿한 마스크)
- Data Engine에서 사람의 역할이 단계별로 어떻게 줄어드는가? 자동 생성 마스크만으로도 성능이 유지되는 이유는?
- SAM은 "무엇인지"를 모르는데, 이를 보완하려면 CLIP 같은 모델과 어떻게 결합할 수 있을까?
