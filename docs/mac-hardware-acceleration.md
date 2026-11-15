# MAC과 하드웨어 가속: CPU SIMD, Pascal CUDA, Tensor Core, NPU 비교

[README의 학습 목표로 돌아가기](../README.md#10-과제-목표-6개-해설)

이 문서는 과제의 MAC 계산을 실제 하드웨어 실행 방식과 연결하는 심화 자료다. 프로그램에 CUDA·AVX·NPU 실행 기능을 추가한 것이 아니다. 제품 벤치마크가 아니라 공식 아키텍처 자료에 근거한 구조 비교이며, 서로 다른 정밀도·세대의 최대 처리량을 실제 속도 향상률처럼 비교하지 않는다.

## 1. 결론부터

CPU와 기존 CUDA GPU도 이미 대량의 MAC을 처리했다. Tensor Core와 NPU의 차별점은 MAC의 발명이 아니라, 행렬·신경망 작업에 맞춰 연산 밀도, 데이터 재사용, 명령·제어 부담과 전력 효율을 최적화한 점이다. [S1][S3][S8]

- CPU SIMD: 한 벡터 명령으로 여러 원소의 FMA를 처리한다.
- Pascal CUDA GPU: 많은 스레드가 여러 SM에서 병렬로 FMA/내적을 처리한다.
- Tensor Core: 행렬 타일 곱·누적을 전용 유닛에서 수행한다.
- NPU: 행렬 연산기뿐 아니라 로컬 메모리·데이터 이동·벡터 연산·실행 제어 등을 포함하는 가속기 범주다.

따라서 CUDA Core, AVX, Tensor Core, NPU는 같은 계층의 제품명이 아니다. CUDA는 프로그래밍 플랫폼이며 Tensor Core를 사용하는 코드 역시 CUDA로 실행할 수 있다. 비교할 대상은 'Pascal의 일반 연산 경로'와 '전용 행렬 연산 경로'다.

## 2. MAC, FMA, MMA와 행렬 곱

MAC: `s ← s + x*w`라는 계산 패턴. FMA: 곱셈과 덧셈을 결합하고 결과를 한 번 반올림하는 연산. MMA: 행렬 블록의 곱·누적. 대표 형태는 `D = A*B + C`다. [S3]

Python의 `score += x*w`가 단일 하드웨어 FMA로 수행된다고 보장할 수 없다. 벡터 FMA 역시 내적 전체를 한 번에 반환하는 명령은 아니다. 각 SIMD lane에서 독립 누적한 뒤 최종 수평 합산이 필요할 수 있다.

패턴을 길이 K=N²의 벡터 p로 펼치고 두 필터를 W의 열에 놓으면:

```text
p: 1×K
W: K×2
scores = pW: 1×2
```

패턴 B개를 쌓으면 `Y = XW`, 즉 `B×K`와 `K×2`의 행렬 곱이다. 이는 수학적 재표현일 뿐 현재 구현을 변경한 것은 아니다. 출력 열이 2개뿐인 작은 문제는 거대한 정방 행렬 곱과 하드웨어 활용률이 다르다.

일반적인 M×K, K×N 행렬 곱의 작업량은 MKN MAC, 곱셈과 덧셈을 각각 세는 관례로 약 2MKN 연산이다. [S7]

## 3. 동시기 CPU와 GPU를 구분해야 한다

GTX 10/Pascal 시기와 비교할 때 모든 Xeon을 AVX-512 CPU로 묶으면 안 된다. Broadwell Xeon E5 v4는 AVX2+FMA를 사용하며, 그 뒤 Skylake Xeon Scalable에서 AVX-512가 도입되었다. [S4]

AMD EPYC도 세대를 나눠야 한다. EPYC 7001은 Zen, 7002는 Zen 2, 7003은 Zen 3이며, AVX-512를 도입한 EPYC Zen 4 세대는 나중의 9004/8004 계열이다. Zen 4 세대의 AVX-512 지원을 초기 EPYC에 소급하면 안 된다. [S5][S6]

| 계열 | 주된 계산 경로 | 구분할 사항 |
|---|---|---|
| Broadwell Xeon E5 v4 | 256-bit AVX 계열 FMA | AVX-512 없음 |
| Skylake Xeon Scalable | AVX-512 FMA | SKU별 FMA 유닛 수·AVX 동작 클럭 확인 |
| EPYC 7001, Zen | AVX2/FMA, 128-bit 실행 경로 기반 | 256-bit ISA와 내부 실행 폭은 별개 |
| EPYC 7002/7003, Zen 2/3 | AVX2/FMA, 256-bit 실행 경로 | Zen 4의 AVX-512와 구분 |
| EPYC 9004/8004, Zen 4/4c | AVX-512 지원 | 내부 256-bit 데이터 경로로 분할 처리 |
| Pascal GP102/104/106 | CUDA FP32 및 DP4A 등 | Tensor Core 없음, FP16 처리량 주의 |
| Pascal GP100 | CUDA FP32/FP64 및 고속 FP16 | Tensor Core 없음, GP104와 다른 설계 |
| Volta GV100 | CUDA 연산 + Tensor Core | 전용 혼합 정밀도 행렬 연산 추가 |

CPU 세대·벡터 폭 근거는 [S4][S5][S6], Pascal과 Volta 구분은 [S1][S2][S3]을 참조한다. 특정 CPU의 ISA와 처리량은 정확한 SKU 및 컴파일 결과로 확인해야 한다.

## 4. CPU AVX2·AVX-512는 어떻게 MAC을 가속하는가?

### 4.1 벡터 폭과 lane 수

FP32 원소를 기준으로 256-bit 벡터에는 8개, 512-bit 벡터에는 16개가 들어간다. FP64라면 각각 4개와 8개다. AVX2와 함께 흔히 사용하는 FMA3는 별도의 ISA 기능이며, 'AVX2라는 이름 자체가 FMA 명령'인 것은 아니다. [S4]

| 벡터 FMA 한 명령 | FP32 원소별 MAC | FP32 연산 수 | FP64 원소별 MAC | FP64 연산 수 |
|---|---:|---:|---:|---:|
| 256-bit | 8 | 16 | 4 | 8 |
| 512-bit | 16 | 32 | 8 | 16 |

위는 명령 한 개의 작업량이지 CPU 코어당 사이클 처리량이 아니다. 사이클당 처리량에는 실행 유닛 수와 명령 발행률을 함께 반영해야 한다.

예를 들어 256-bit FMA를 2개/사이클 처리하면 FP32 최대 32연산/사이클, 512-bit FMA를 2개/사이클 처리하면 최대 64연산/사이클이다. 이는 충분한 독립 연산과 데이터가 공급된다는 이상적 조건의 설명이다. [S4]

### 4.2 왜 AVX-512가 항상 두 배 빠르지는 않은가?

- CPU 모델에 따라 FMA 실행 유닛 개수가 다르다.
- 넓은 벡터 연산 시 동작 클럭과 전력 조건이 달라질 수 있다.
- 메모리와 캐시가 연산 유닛에 데이터를 공급하지 못할 수 있다.
- 누적합 하나에 의존하면 FMA 지연시간 때문에 처리량을 다 쓰지 못한다.
- 레지스터·타일 재사용과 여러 독립 누적값이 필요하다.
- 작은 입력은 벡터 준비·나머지 처리·호출 비용의 비중이 크다. [S4]

Zen 4는 512-bit 레지스터를 지원하지만 내부에는 256-bit 데이터 경로를 사용한다. 따라서 ISA 지원 여부와 물리적인 한 사이클 실행 폭을 같은 것으로 보면 안 된다. [S5]

### 4.3 Python 프로그램을 그대로 실행하면 AVX2가 사용되는가?

현재의 중첩 리스트와 Python 객체 연산은 배열 전체를 자동으로 AVX2/AVX-512 FMA 루프로 바꾸는 구현이 아니다. 하드웨어 가속 비교를 하려면 연속된 float/double 배열과 C/C++ 벡터화, 또는 BLAS 등의 최적화 경로가 별도로 필요하다. 현재 과제는 외부 라이브러리를 금지하므로 이 실험은 제출 코드와 분리해야 한다.

## 5. GTX 10·Pascal Quadro의 CUDA 가속은 어떻게 다른가?

### 5.1 Tensor Core가 없어도 AI 계산은 가능하다

Pascal은 SM의 워프 스케줄러와 다수의 일반 연산 유닛으로 병렬 계산한다. 행렬 타일을 레지스터와 shared memory에 두고 재사용하는 최적화도 이미 사용한다. 따라서 '예전 GPU는 데이터를 매 MAC마다 VRAM에서 다시 읽었다'는 설명은 틀리다. [S1]

새 Tensor Core의 장점은 데이터 재사용을 처음 가능하게 한 것이 아니라, 기존 소프트웨어 타일링과 결합할 전용 행렬 계산 경로를 추가한 것이다.

### 5.2 Pascal이라고 FP16 성능이 같지는 않다

NVIDIA 공식 가이드에서 GP104의 native FP16 처리량은 FP32의 1/64인 반면, GP100은 half2를 사용하는 조건에서 최대 FP32의 2배 FP16 처리량을 제공한다. [S1]

| 비교 | GP104 계열 | GP100 |
|---|---|---|
| 설계 성격 | 그래픽/FP32 및 정수 추론 경로 강조 | 과학 계산·딥러닝용 FP64/FP16 강화 |
| native FP16 | FP32 대비 1/64 | 적합한 half2 경로에서 최대 2배 |
| Tensor Core | 없음 | 없음 |
| DP4A | 지원 | 미지원 |

FP16 저장은 메모리 사용량을 줄일 수 있지만, FP16 산술이 무조건 FP32보다 빠르다는 뜻은 아니다. 라이브러리가 FP16 입력을 어떻게 계산하는지도 별도로 확인해야 한다. [S1]

Quadro라는 브랜드만으로 FP16·FP64·Tensor Core 능력을 판단하면 안 된다. 실제 Quadro GP100은 빠른 FP16을 강조하는 Pascal 제품이다. [S10] GTX 10 시기 Quadro를 비교할 때는 정확한 GPU 칩과 세대를 확인해야 한다.

### 5.3 기존에도 INT8 내적 가속이 있었다

GP102/GP104/GP106의 DP4A는 8-bit 정수 4쌍의 곱을 INT32에 누적한다. 즉, 단일 lane 수준의 짧은 정수 내적도 Tensor Core 이전부터 지원했다. [S2]

```text
d = c + a0*b0 + a1*b1 + a2*b2 + a3*b3
```

따라서 'INT8 가속이 있으니 Tensor Core다'라는 판단도 틀리다. DP4A는 짧은 벡터 내적이며, 행렬 타일 전용 MMA와 같은 단위가 아니다.

이 문서의 Pascal 설명은 당시 하드웨어 능력에 대한 것이다. 현재 CUDA·드라이버·프레임워크가 해당 GPU를 계속 지원하는지는 별도의 호환성 문제다.

## 6. Tensor Core가 추가한 것

### 6.1 벡터/스레드별 FMA에서 행렬 타일 MMA로

Volta GV100은 일반 FP32/FP64/INT32 유닛과 함께 SM당 8개 Tensor Core를 갖는다. 초기 Tensor Core는 FP16 입력의 4×4 행렬 곱·누적을 수행하고, 누적은 FP16 또는 FP32 모드를 제공한다. [S3]

초기 Tensor Core 한 개의 작업량 설명은 다음과 같다.

```text
출력 4×4 = 16개
각 출력의 내적 길이 = 4
→ 64개의 원소별 곱·누적
→ 곱셈·덧셈 별도 계산 시 128연산
```

NVIDIA가 제시하는 64 FMA/클럭은 이 전용 유닛의 구조적 처리량이다. [S9] CPU 코어나 CUDA Core와 면적·전력·정밀도·유닛 수가 같지 않으므로 'Tensor Core 한 개가 CPU 코어보다 64배 빠르다'라고 해석할 수 없다. 또한 이 수치는 초기 Volta의 예이며 최신 세대의 tile 구조와 동일하다고 가정하지 않는다.

### 6.2 CUDA를 대체한 것이 아니다

CUDA는 실행·프로그래밍 플랫폼이다. Tensor Core는 CUDA 라이브러리나 WMMA/MMA 경로에서 사용할 수 있다. 같은 CUDA 프로그램 안에서 Tensor Core 경로와 일반 연산 경로를 조합할 수 있다. [S3]

### 6.3 무조건 더 빠르지 않은 이유

행렬 크기, 정렬, tile 활용률, 지원 정밀도와 라이브러리의 경로 선택이 중요하다. 계산 전에 형식을 바꾸거나 데이터를 옮기는 비용도 포함해야 한다. 정밀도 변경과 누적 순서 때문에 기존 float64 코드와 bitwise 동일 결과를 보장할 수 없다. [S7]

## 7. NPU는 무엇을 더 특화하는가?

NPU는 신경망 처리용 가속기 범주이며 Tensor Core라는 특정 GPU 유닛과 동의어가 아니다. 행렬/벡터 계산기, 로컬 메모리, 데이터 이동·실행 제어 등을 타깃 신경망 실행에 맞춰 설계하는 것이 핵심이다.

구체적인 예로 Google TPU는 ML용 ASIC으로, MXU의 시스톨릭 배열에서 곱셈·누적을 수행하며 벡터·스칼라 유닛을 함께 갖는다. [S8] 시스톨릭 배열은 데이터를 인접 계산기로 전달하면서 재사용하는 구조의 예다. 모든 NPU가 같은 시스톨릭 구조나 같은 정밀도·전력 목표를 갖는 것은 아니다.

가능한 이점:
- 지원되는 그래프에서 높은 연산 밀도와 데이터 재사용
- 범용 명령·스케줄링 부담 감소
- 적합한 저정밀도와 로컬 메모리를 통한 에너지·대역폭 절감

대가와 제한:
- 지원 연산·shape·정밀도 및 컴파일러에 의존
- 미지원 연산의 CPU fallback과 장치 간 동기화 비용
- 작은 입력 또는 메모리 병목 작업에서 낮은 활용률
- 그래프 변환·양자화에 따른 정확도 재검증 필요

이득의 크기는 특정 제품과 실제 워크로드로 측정해야 한다. 저전력 클라이언트 NPU와 데이터센터용 가속기를 하나의 전력 등급으로 묶으면 안 된다.

## 8. 왜 행렬 크기와 배치가 중요한가?

단일 FP32 내적을 두 벡터에서 각각 한 번 읽는 이상적 모델에서는 원소당 2연산과 8바이트 읽기이므로 산술집약도는 약 0.25연산/바이트다. 이는 출력·캐시·재사용 등을 단순화한 모델이며 현재 Python 객체 메모리 모델의 수치가 아니다.

한편 행렬 곱은 같은 A/B 원소를 여러 출력에 재사용할 수 있어 잘 타일링하면 연산 대비 데이터 이동을 줄일 수 있다. [S7]

전용 연산기가 아무리 빨라도 데이터를 공급받지 못하면 놀게 된다. 그래서 높은 TOPS만 보지 말고 메모리 대역폭·SRAM/캐시·tiling·batch·전송을 함께 본다.

대략적인 지연시간 분해:

```text
전체 지연 = 준비/변환 + 전송 + 실행 요청 + 커널 계산 + 동기화/결과 회수
```

- CPU: 작은 작업이나 이미 CPU 메모리에 있는 데이터에서 유리할 수 있다.
- Pascal GPU: 충분한 병렬 작업과 resident 데이터가 있으면 기존 FMA 경로도 강력하다.
- Tensor Core GPU: 큰 호환 행렬 작업에서 전용 MMA 처리량을 활용한다.
- NPU: 지원 그래프와 배포 조건에서 실행·에너지 효율을 목표로 한다.

## 9. 이번 과제에 적용하면?

최대 25×25, 필터 2개의 한 패턴 판정은 1250 MAC이다. 수치상 작은 작업이므로 가속기 전송·호출 비용과 비교해야 한다. 현재 프로그램의 약 수십 마이크로초 결과는 Python 반복문 비용을 포함한다. 이 값과 GPU의 TFLOPS를 바로 나눠서 실측 가속률로 삼으면 안 된다.

제출 코드는 표준 라이브러리와 반복문을 유지한다. 별도 심화 실험에서는 다음 조건을 맞추는 것이 공정하다.

1. Python 기준 구현과 CPU 최적화 구현을 분리한다.
2. 같은 정밀도의 CPU SIMD와 GPU 일반 FMA를 먼저 비교한다.
3. 낮은 정밀도 Tensor Core/NPU는 별도 정확도 조건으로 비교한다.
4. 단일 내적, 필터 2개, 큰 batch, 큰 GEMM을 구분한다.
5. 장치 내부 계산 시간과 전송 포함 end-to-end 시간을 모두 기록한다.
6. 준비 실행·동기화·타이머·클럭·스레드 수·라이브러리 버전을 고정한다.
7. 결과 오차와 UNDECIDED 변화를 함께 보고한다.
8. 실제 가속기 사용 여부는 profiler나 assembly로 확인한다.

특히 이번 동점 데이터는 정밀도를 바꾸면 아주 작은 수치 차이로 가짜 승자가 생길 수 있다. 성능이 높아졌다는 이유로 정답 판정이 더 정확해졌다고 단정하지 않는다.

## 10. 오해 정리

| 오해 | 정확한 구분 |
|---|---|
| CPU는 직렬이고 GPU만 병렬이다 | CPU도 SIMD·멀티코어·FMA를 사용한다 |
| Tensor Core 없으면 AI 계산 불가 | Pascal CUDA와 CPU도 가능하다 |
| Quadro면 FP16이 빠르다 | 실제 칩·아키텍처·모델을 확인해야 한다 |
| FP16 저장을 하면 연산도 두 배 빠르다 | 저장과 산술 처리량은 다르다 |
| AVX-512면 모든 CPU가 같은 속도다 | 내부 실행 폭·유닛 수·클럭이 다르다 |
| CUDA와 Tensor Core는 경쟁 플랫폼이다 | CUDA는 플랫폼, Tensor Core는 유닛이다 |
| 모든 NPU는 Tensor Core와 같다 | 분류 계층과 구조가 다르다 |
| 1 MAC은 항상 스펙상 1연산이다 | FLOPS/TOPS에서 곱셈·덧셈을 별도 계산하는지 확인한다 |
| TFLOPS가 높으면 작은 과제도 빠르다 | 호출·전송·활용률·메모리 병목을 확인한다 |

## 11. 출처와 적용 범위

[S1] NVIDIA, Pascal Tuning Guide — GP100/GP104, FP16, SM, 메모리와 타일링 관련 근거.
https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html

[S2] NVIDIA, Mixed-Precision Programming with CUDA 8 — GP102/104/106 DP4A 및 Pascal 정밀도 차이.
https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/

[S3] NVIDIA, Volta Tuning Guide — Tensor Core와 WMMA, 4×4 MMA 및 정밀도.
https://docs.nvidia.com/cuda/archive/13.1.2/volta-tuning-guide/index.html

[S4] Colfax Research, Capabilities of Intel AVX-512 in Xeon Scalable (2017) — Broadwell/Skylake 벡터화, FMA, SKU·클럭·메모리 조건.
https://colfaxresearch.com/skl-avx512/

[S5] AMD, 4th Gen EPYC Processor Architecture — EPYC 세대 구분과 Zen 4의 AVX-512/256-bit 내부 경로.
https://docs.amd.com/api/khub/documents/Fk9hEabDNdOiD4PcQu7EYw/content

[S6] AMD, Zen Core Architecture — 세대별 구조, Zen 2의 256-bit FP 처리량 확장.
https://www.amd.com/en/technologies/zen-core.html

[S7] NVIDIA, Matrix Multiplication Background User's Guide — FLOP 산정, tiling과 메모리 병목.
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

[S8] Google Cloud, TPU architecture — ASIC, MXU·시스톨릭 배열, 벡터·스칼라 유닛.
https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm

[S9] NVIDIA, Tensor Cores in Volta — 초기 Tensor Core당 64 FMA/클럭.
https://www.nvidia.com/en-gb/data-center/tensorcore/

[S10] NVIDIA, Quadro GP100 Data Sheet — Pascal 전문가용 제품의 FP16 특성 예.
https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-pascal-gp100-data-sheet-us-nv-704562-r1.pdf

실측하지 않은 제품 간 배수나 전력 수치는 제시하지 않았다. 이 문서의 비교는 아키텍처 원리이며 실제 성능은 정확한 SKU와 소프트웨어 조건에 따라 달라진다.
