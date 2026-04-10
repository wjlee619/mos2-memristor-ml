# MoS₂ Memristor ML Project — Learning Notes
> Jun의 프로젝트 학습 노트. 각 Phase에서 배운 개념, 수식, 판단 이유, 문제와 해결 방법 정리.
> GitHub 공개 문서.

---

## 프로젝트 큰 그림

```
실험 데이터 (CSV 파일들)
        ↓ Phase 1 + Phase 1b + Phase 1c
Feature Matrix (숫자 표)
        ↓ Phase 2 (EDA) ✅
패턴 시각화 + 이상치 확인
        ↓ Phase 3 (ML) ✅
Random Forest → ON 상태 안정성 분석
```

**핵심 질문:** 레이어 수 + 측정 조건 → 전기 특성 예측 가능한가?

**최종 발견:** 레이어 수만으로는 예측 불가. 반복 측정에 따른 Electroforming 효과가 지배적 변수.

---

## 데이터 구조

| 폴더 | 파일 수 | 소자 타입 | Phase | 상태 |
|------|---------|-----------|-------|------|
| `probe/` | 73개 | 트랜지스터 (gate sweep) | Phase 1 | ✅ |
| `15082024/` | 39개 | 멤리스터 (IV sweep) | Phase 1b | ✅ |
| `2024-08-30/` | 85개 CSV | 멤리스터 (IV sweep) | Phase 1c | ✅ |
| `uv vis/` | 8개 | UV-Vis 흡광도 | 추후 | 예정 |

---

## Phase 1 — 트랜지스터 Gate Sweep ✅

### 배경: 왜 이 데이터부터 시작했나

MoS₂ 소자의 기본 반도체 특성을 먼저 파악해야 ML feature로 쓸 수 있는 숫자를 추출할 수 있어. Gate sweep은 "이 소자가 트랜지스터로서 얼마나 잘 작동하나"를 보여주는 가장 기본적인 측정이야.

### 핵심 개념

**트랜지스터 (Transistor)**
- 전기로 제어하는 스위치
- 세 단자: Gate (제어), Drain (입력), Source (출력)
- Gate 전압(Vgs)이 threshold 이상 → 채널 형성 → 전류 흐름 (ON)
- Gate 전압이 낮음 → 채널 없음 → 전류 차단 (OFF)

**Gate Sweep**
- Vgs를 –20V → +20V로 천천히 올리면서 Drain 전류(Id)를 측정
- CSV 한 파일 = 실험 한 번
- MoS₂ 레이어 수별로 반복 측정 → 레이어 두께가 특성에 미치는 영향 파악

### 추출한 Features와 이유

| Feature | 수식 | 의미 | 좋은 값 |
|---------|------|------|---------|
| `id_on_A` | max\|Id\| | ON 상태 전류 | 클수록 좋음 |
| `id_off_A` | min\|Id\| (0 제외) | OFF 상태 전류 | 작을수록 좋음 |
| `on_off_ratio` | id_on ÷ id_off | 스위치 성능 지표 | 클수록 좋음 |

**왜 이 세 가지냐:** 트랜지스터 성능을 하나의 숫자로 요약하는 업계 표준 지표이기 때문. 논문에서도 항상 이 값들로 소자를 비교해.

### 발생한 문제와 해결

**문제 1: Python 아키텍처 충돌**
```
에러: numpy가 x86_64용으로 설치됐는데 Mac은 ARM64
원인: /usr/local/bin/python3 (Intel용) 사용
해결: /opt/homebrew/bin/python3.12 (ARM64 네이티브)로 고정
```

**문제 2: Noise Floor**
```
발견: id_off가 전부 ~3.66×10⁻⁶ A로 동일
원인: Keithley 장비의 측정 분해능 한계 (noise floor)
의미: 실제 on/off ratio는 5~8보다 높을 수 있음
대응: ML feature로 id_off 쓸 때 이 한계를 명시
```

### 주요 발견
- 73개 파일, 6개 레이어 그룹: 10, 20, 30, 40, 50, 60 layers
- 40L에서 on/off ratio 16x 이상치(outlier) 존재

---

## Phase 1b — 멤리스터 IV Sweep ✅

### 배경: 왜 Phase 1 다음에 했나

Gate sweep은 트랜지스터 특성을 봤어. 근데 이 소자의 핵심은 **멤리스터** — 메모리 특성이야. SET/RESET voltage를 추출해야 "어떤 조건에서 스위칭이 일어나나"를 ML로 분석할 수 있어.

### 핵심 개념

**멤리스터 (Memristor)**
- Memory + Resistor
- 전원을 꺼도 ON/OFF 상태를 기억하는 소자
- 차세대 비휘발성 메모리 (뇌의 시냅스와 유사한 동작)

**IV Sweep**
- 전압을 0 → ±20V → 0으로 한 바퀴 돌리면서 전류 측정
- SET 순간: 전류가 갑자기 확 뜀 (OFF→ON)
- RESET 순간: 전류가 갑자기 확 꺼짐 (ON→OFF)

**쌍극성 멤리스터 (Bipolar Memristor)**
- 양전압으로도 SET, 음전압으로도 SET 가능
- Run 33 (음전압 –20V)과 Run 35 (양전압 +20V) 둘 다 스위칭 확인

**Hysteresis Window**
- \|SET voltage – RESET voltage\|
- 클수록 메모리 안정성 높음
- Run 33: 9.46V / Run 35: 17.99V

### SET/RESET 감지 알고리즘 — 왜 이 방법을 썼나

**시도 1: 단순 dI/dV (실패)**
```
문제: OFF 전류 ~10⁻¹² A, ON 전류 ~10⁻⁵ A → 범위가 10⁷배
결과: ON 상태 근처의 작은 노이즈도 크게 보여서 SET 지점 못 잡음
```

**해결: d(log₁₀|I|)/dV**
```
아이디어: 전류를 log scale로 변환 → 범위가 –12 ~ –5로 압축
원리: SET이 일어날 때 log 전류가 가장 가파르게 변함
```

수식:
```
log_i = log₁₀(|I|)
grad = d(log_i)/dV = [log_i(V+ΔV) - log_i(V)] / ΔV
SET voltage = argmax(grad)   # 가장 가파르게 올라가는 지점
RESET voltage = argmin(grad) # 가장 가파르게 내려가는 지점
```

코드:
```python
log_i = np.log10(np.abs(current))    # log 변환
grad_idx = np.gradient(log_i)         # 기울기 계산 (index 기준)
dv = np.gradient(np.abs(voltage))     # 전압 변화량
grad_v = grad_idx / np.abs(dv)        # per-volt 변환 (decades/V)
switch_idx = np.argmax(grad_v)        # SET = 최댓값
# RESET = np.argmin(grad_v)
```

**왜 smoothing을 추가했나:**
```python
kernel = np.ones(20)/20
grad_smooth = np.convolve(grad_v, kernel, mode='same')
```
노이즈 때문에 기울기가 들쭉날쭉 → 20포인트 이동 평균으로 부드럽게 만든 후 최댓값 탐색

### switching_state 분류 로직 — 왜 필요했나

39개 파일 중 실제 스위칭이 2개뿐인 이유를 이해해야 데이터를 올바르게 해석할 수 있어.

```
시작 전류가 이미 크다? (median(I_초기) > i_on × 0.01)
        ↓ Yes → already_on   (이전 Run에서 SET된 상태 유지)
        ↓ No
최대 전압 < 2V?
        ↓ Yes → low_voltage_sweep  (진단용 저전압 측정)
        ↓ No
d(log|I|)/dV 최댓값 > threshold?
        ↓ Yes → switched           (실제 스위칭 감지)
        ↓ No  → no_switching_detected
```

**already_on이 31개인 이유:**
연속 측정 세션에서 Run 33이 SET되면 Run 36, 37, 38...은 이미 ON 상태에서 시작. 데이터 품질 문제가 아니라 실험 설계 특성.

### 주요 발견

| | Run 33 (–20V) | Run 35 (+20V) |
|-|---------------|---------------|
| V_SET | –12.83 V | +19.41 V |
| V_RESET | –3.36 V | +1.42 V |
| I_ON | 16 μA | 9.5 μA |
| I_OFF | ~1 pA | ~0.4 pA |
| ON/OFF ratio | ~1.5×10⁷ | ~2.6×10⁷ |
| Hysteresis | 9.46 V | 17.99 V |

### 논문 수치 vs 추출 수치

| | MRes 논문 | Phase 1b 추출 |
|-|-----------|---------------|
| SET voltage | ~9V | –12.83V |
| RESET voltage | ~–13V | –3.36V |
| on/off ratio | 1.366 | ~1.5×10⁷ |

> **차이 이유:** 논문의 1.366은 선형 전류 비율 (단위: 배), 코드는 절대 전류값 비율. 측정 방법이 다름.

---

## Phase 1c — 2024-08-30 데이터 ✅

### 배경: 왜 추가했나

Phase 3 ML에서 데이터가 73개(probe)밖에 없어서 모델 신뢰도가 낮았어. 2024-08-30/ 폴더에 85개 CSV가 있어서 데이터를 늘리려고 시도.

### 발생한 문제

**문제: DeviceDatabase.mat 파싱 실패**
```
시도: scipy.io.loadmat()으로 읽기
에러: MATLAB MCOS 포맷 — scipy가 직접 디코딩 불가
결론: MATLAB이 없으면 파싱 불가 → 포기하고 CSV로 진행
```

**문제: switched 데이터 부족**
```
기대: FB4-T12 (18 runs)에서 스위칭 다수 발견
실제: switched 1개 (Run 81, 의심스러운 값)
원인: 대부분 already_on — 이전 세션에서 SET된 상태 장기 유지
의미: 멤리스터의 비휘발성 메모리 특성 증명
```

### 결론
- 85개 파일 중 switched 1개 (V_RESET > V_SET → 알고리즘 오감지 가능성)
- 전략 변경: SET voltage 예측 → ON 상태 안정성 분석

---

## Phase 2 — EDA ✅

### EDA란?
> Exploratory Data Analysis — ML 모델 돌리기 전에 데이터를 눈으로 먼저 이해하는 과정.
> 의사가 수술 전에 X-ray 찍는 것과 같음. 데이터에 문제가 있어도 ML은 그냥 돌아가지만 결과가 쓰레기가 됨.

### EDA에서 확인하는 4가지

| 항목 | 의미 | 도구 |
|------|------|------|
| **분포** | 각 feature 값이 어떻게 퍼져있나 | Histogram, Boxplot |
| **상관관계** | 두 feature가 같이 움직이나 | Correlation Heatmap |
| **이상치** | 말도 안 되는 값이 있나 | Scatter plot |
| **데이터 품질** | NaN이 얼마나 있나 | 결측값 테이블 |

### Section 1 — 레이어 분포

**발견:** 레이어 수 ↔ 전기 특성 상관계수 r < 0.25

**상관계수(r)란:**
```
r = +1.0 → 완벽한 양의 관계
r =  0.0 → 관계 없음
r = -1.0 → 완벽한 반대 관계
r < 0.25 → 선형 관계 없음
```

**의미:** 레이어 수만으로 on/off ratio 예측 어려움 → ML에서 더 복잡한 패턴 탐색 필요

### Section 3 — 상관관계에서 발견한 Spurious Correlation

**Spurious Correlation (허위 상관)이란:**
> 진짜 물리적 관계가 아니라 계산 방법 때문에 생긴 가짜 상관관계

**우리 케이스:**
```
on_off_ratio = i_on ÷ i_off
i_off = 전부 3.66×10⁻⁶ A (noise floor 고정값)
→ on_off_ratio = i_on ÷ (상수) = i_on × 상수
→ i_on과 on_off_ratio의 r = 0.96 (강한 상관)
```

**왜 문제냐:**
- 같은 정보를 두 번 ML에 넣는 것 = **Data Leakage**
- Feature Importance가 왜곡됨
- **해결:** i_on을 feature로, on_off_ratio를 target으로만 사용

### Section 4 — 데이터 품질

- v_set_V, v_reset_V: **94.9% NaN** → switched 2개만 유효값
- **ML 제약:** SET voltage를 target으로 쓰는 ML 불가 → 전략 변경 필요

---

## Phase 3 — ML ✅

### 왜 Random Forest를 선택했나

**옵션 1 — 선형 회귀 (탈락)**
```
이유: EDA에서 r<0.25 → 직선 관계 없음 → 선형 모델 부적합
```

**옵션 2 — Deep Learning (탈락)**
```
이유: 데이터 73개 → 수만 개 필요한 신경망 부적합 → overfitting
```

**옵션 3 — Random Forest (선택)**
```
이유 1: 작은 데이터에 강함 (73개도 가능)
이유 2: 비선형 패턴 감지 가능
이유 3: Feature Importance 자동 계산
이유 4: Overfitting에 상대적으로 강함 (500개 트리 평균)
```

### Random Forest 작동 원리

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)      # 트리 500개 학습 (자동)
y_pred = model.predict(X_test)   # 500개 예측의 평균
model.feature_importances_       # Feature Importance (자동 계산)
```

**n_estimators=500:** 트리 500개. 많을수록 안정적이지만 속도 느림. 500이 표준.
**random_state=42:** 결과 재현성을 위해 랜덤 시드 고정. 42는 관례.

### Train/Test Split — 왜 나누나

```
전체 73개 → Train 58개 (80%) + Test 15개 (20%)
```

**이유:** 모델이 "답을 외우는 것(Overfitting)"을 방지.
Test 데이터는 학습에 쓰지 않고 나중에 "처음 보는 시험"으로 사용.

### 평가 지표

**R² (결정계수):**
```
R² = 1 - (예측 오차² 합) / (실제값 분산)
R² = 1.0 → 완벽한 예측
R² = 0.0 → 평균값 찍는 것과 같음
R² < 0   → 평균값보다 더 못함
```

**RMSE (Root Mean Square Error):**
```
RMSE = √(Σ(실제값 - 예측값)² / n)
→ 예측 오차의 평균 크기 (단위: target과 같음)
```

### Phase 3a 결과 — on/off ratio 예측 (실패, 하지만 중요한 발견)

| 지표 | Train | Test |
|------|-------|------|
| R² | 0.152 | –0.087 |
| RMSE | 2.116 | 1.063 |
| 5-fold CV R² | –0.425 ± 0.471 | — |

**Feature Importance 역설:**
```
layers importance = 87.7%
→ 모델이 레이어 수를 제일 중요하게 여김
→ 그런데 Test R² = –0.087 → 실제 예측은 실패
```

**해석:**
> 레이어 수는 on/off ratio와 통계적으로 무관하다.
> EDA의 r<0.25를 ML이 수학적으로 재확인한 것.

**포트폴리오 표현:**
> *"Random Forest 분석 결과 레이어 수만으로는 on/off ratio 예측 불가 (R²=–0.09) — 측정 배치 효과나 공정 조건이 더 지배적인 변수임을 시사"*

### Phase 3b — 전략 변경: ON 상태 안정성 분석

**왜 전략을 바꿨나:**
```
switched 데이터: 15082024에서 2개, aug30에서 1개 (의심스러운 값)
→ SET voltage를 target으로 쓰는 ML 불가
→ already_on 115개가 오히려 안정성 분석에 완벽한 데이터
```

**새 질문:**
> "배치(Chip)별, Run별로 ON 상태 전류(i_on)가 얼마나 일관성 있게 유지되나?"

**분석 도구: Mann-Whitney U 검정**
```
용도: 두 그룹의 분포가 유의미하게 다른지 검정
H₀ (귀무가설): 두 그룹의 중앙값이 같다
p < 0.05 → H₀ 기각 → 유의한 차이 있음
p ≥ 0.05 → H₀ 채택 → 유의한 차이 없음
```

왜 t-test가 아니라 Mann-Whitney냐:
```
t-test 가정: 데이터가 정규분포를 따름
우리 데이터: i_on이 log scale로 넓게 분포 → 정규분포 아님
Mann-Whitney: 정규분포 가정 없음 → 더 적합
```

### Phase 3b 결과

**Chip별 i_on 분포:**

| Chip | n | 중앙값 | std |
|------|---|--------|-----|
| Chip#1 | 31 | 2.94mA | 2.23mA |
| Chip#14 | 81 | 6.05mA | 4.22mA |
| Chip#6 | 3 | 2.65pA | — |

Chip#6이 pA 수준인 이유: 제대로 SET이 안 된 소자 (low_voltage_sweep과 유사)

**Run별 추세 — 가장 중요한 발견:**
```
Chip#14: slope > 0, R²=0.478, p<0.001
→ Run 번호가 올라갈수록 i_on이 유의하게 증가
```

**Electroforming 효과:**
> 멤리스터를 반복 측정할수록 전도성 필라멘트(conductive filament)가 점점 안정적으로 형성됨.
> 초기 몇 번의 sweep이 소자를 "훈련"시키는 효과.
> Chip#14에서 이 효과가 데이터로 확인됨 (R²=0.48, p<0.001).

**광조건 영향:**
```
Dark-ThenLight vs Other: p=0.25 → 유의한 차이 없음
해석: 이미 ON 상태에서는 빛이 전류에 영향 없음
      스위칭 순간에는 영향을 줄 수 있지만 안정된 ON 상태는 빛에 무관
```

**포트폴리오 표현:**
> *"Chip#14에서 반복 측정에 따른 ON 전류 증가 추세 확인 (R²=0.48, p<0.001) — 전도성 필라멘트 형성(electroforming) 과정과 일치. 광조건(Dark vs Light)은 이미 ON 상태의 전류에 유의한 영향 없음 (p=0.25)."*

---

## 전체 프로젝트 스토리

```
Phase 1  → 레이어 수만으로 성능 예측 불가 확인 (r<0.25)
                ↓
Phase 1b → 쌍극성 스위칭 확인 (±전압 모두 SET 가능)
           ON/OFF ratio ~10⁷ (절대값 기준)
                ↓
Phase 1c → 데이터 확장 시도 → switched 여전히 극소수
           전략 변경 결정
                ↓
Phase 2  → EDA로 Spurious Correlation 발견 + 데이터 품질 파악
                ↓
Phase 3a → Random Forest로 레이어 수의 무관함 수학적 확인
                ↓
Phase 3b → Electroforming 효과 발견 (R²=0.48)
           광조건은 ON 상태에 무관함 확인
```

---

## 기술 환경 이슈 및 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| numpy ImportError | x86_64 numpy가 ARM64 Mac에서 실행 | `/opt/homebrew/bin/python3.12` 고정 |
| pip install 실패 | Homebrew externally-managed-environment | `--break-system-packages` 플래그 추가 |
| nbformat 없음 | 패키지 미설치 | `pip3.12 install nbformat --break-system-packages` |
| MATLAB .mat 파싱 실패 | MCOS 포맷은 scipy 미지원 | CSV 데이터로 대체 진행 |

---

## 기술 용어 빠른 참조

| 용어 | 한 줄 설명 |
|------|-----------|
| Vgs | Gate-Source 전압 |
| Id | Drain 전류 |
| on/off ratio | ON 전류 ÷ OFF 전류. 클수록 좋은 스위치 |
| SET voltage | 멤리스터가 OFF→ON 되는 전압 |
| RESET voltage | 멤리스터가 ON→OFF 되는 전압 |
| Hysteresis | SET-RESET 전압 차이. 클수록 메모리 안정적 |
| Bipolar memristor | 양/음 전압 방향 모두 스위칭 가능한 멤리스터 |
| Electroforming | 반복 측정으로 전도성 필라멘트가 안정화되는 현상 |
| Noise floor | 장비가 측정할 수 있는 최솟값 한계 |
| Feature | ML 모델에 입력하는 숫자 하나 |
| Target | ML 모델이 예측하는 숫자 |
| EDA | 데이터를 시각화로 먼저 이해하는 과정 |
| Feature Importance | 각 feature가 예측에 기여하는 정도 |
| Outlier | 다른 값들과 동떨어진 이상치 |
| Correlation (r) | –1~+1. 두 feature가 함께 변하는 정도 |
| Spurious Correlation | 계산 방법 때문에 생긴 가짜 상관관계 |
| Data Leakage | 예측하려는 답이 feature 안에 이미 들어있는 문제 |
| log scale | 넓은 범위의 값을 압축해서 보는 방법 |
| d(log\|I\|)/dV | 전압 대비 log 전류 변화율. SET 감지에 사용 |
| Smoothing | 노이즈 제거를 위한 이동 평균 |
| NaN | 측정 실패 또는 해당 없음을 나타내는 결측값 |
| Overfitting | 모델이 학습 데이터를 외워서 새 데이터에서 실패 |
| Train/Test Split | ML 학습용과 검증용 데이터 분리 |
| Cross-validation | 여러 번 나눠서 모델 신뢰도 검증 |
| R² | 모델 설명력. 1=완벽, 0=평균 수준, <0=평균보다 못함 |
| RMSE | 예측 오차의 평균 크기 |
| Mann-Whitney U | 정규분포 가정 없는 두 그룹 비교 검정 |
| p-value | 결과가 우연일 확률. <0.05이면 유의한 차이 |
| Linear Regression | 직선 관계를 찾는 가장 단순한 ML 모델 |
| Random Forest | Decision Tree 500개의 다수결. 소규모 데이터에 강함 |
| Decision Tree | 스무고개처럼 가지치기로 예측하는 모델 |

---

## 다음 단계 (예정)

- [ ] Phase 4: 2024-08-30 데이터 Electroforming 분석 심화
- [ ] Phase 5: UV-Vis 흡광도 데이터 통합
- [ ] GitHub README 작성
- [ ] arXiv preprint 준비

---

*마지막 업데이트: Phase 3 완료 (2026-04-10)*
