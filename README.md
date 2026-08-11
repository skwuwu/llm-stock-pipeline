# LLM Stock Pipeline (KR)

한국 상장사를 **정량 스크린 → LLM 테마 분류 → 검증 레이어 → 촉매 추출**로 훑어
사람이 읽을 리서치 다이제스트를 만든다.

이 시스템의 목적은 **종목 추천이 아니다.** 투자 아이디어를 가진 사람이
"이 가설에 맞는 종목이 실제로 있는가, 그 근거는 원문에 있는가"를 30초 안에
확인할 수 있게 하는 **필터이자 감시망**이다. 최종 판단은 사람이 한다.

```
[가설] "전력망 쇼티지 수혜주를 재무 위험 없이 골라보고 싶다"
   │
   ├─ 정량 스크린      업종·규모·유동성·재무건전성으로 후보를 좁힌다
   ├─ LLM 테마 분류    사업보고서 원문에서 "이 회사가 무엇을 하는가"를 뽑는다
   ├─ 검증 레이어      인용이 원문에 실제로 있는가 / 매출비중이 하한을 넘는가
   ├─ 촉매 추출        "왜 지금인가" — 공시된 사실만. 자사주·수주·실적·희석
   └─ 다이제스트       테마 / 촉매 / 비중근거 / 재무위험 네 축을 나란히 제시
```

---

## 이 시스템이 다른 점

### 1. LLM이 투자 판단을 하지 않는다

LLM에 **주가·PER·PBR·시총을 넘기지 않는다.** evidence pack의 지표는 네 개뿐이다:

```python
METRIC_KEYS = ["revenue_ttm", "operating_income_ttm", "net_income_ttm", "sector_code"]
```

시스템 프롬프트 7번 규칙: *"주가, 밸류에이션, 투자 매력도, 목표주가, 매수/매도
의견을 언급하지 않는다."* LLM의 산출물은 **사실 서술**이다 —
`"CNC선반·밀링 등 공작기계 제조가 주력 사업"`. "저평가됐다"가 아니다.

이유는 셋이다. (a) 투자 의견은 원문으로 대조할 수 없어 **검증이 불가능**하고,
(b) 정답이 없어 **오분류율을 잴 수 없으며**, (c) 밸류에이션을 넣으면 주가가
바뀔 때마다 캐시가 깨져 매일 전 종목 재호출이 된다.

### 2. 측정 가능한 오분류율

사람이 라벨링한 골든셋 **126종목**(`tests/golden/`)에 대해 매 실행 정밀도·재현율을
계산한다. 라벨 규칙은 `tests/golden/LABELING.md`에 명문화돼 있다.

| 스크린 | 오분류율 | certain 라벨만 | 인용 환각률 |
|---|---:|---:|---:|
| deep_value | 3.9% | 0.0% | 1.16% |
| garp | 3.1% | 0.0% | 0.00% |
| quality_fcf | 5.7% | 0.0% | 1.67% |

### 3. 재현성을 재고, 그 수치를 리포트에 싣는다

LLM 파이프라인은 완전히 결정적이지 않다. 그래서 **얼마나 흔들리는지 측정해
공개한다** (`scripts/determinism_probe.py`, 캐시 우회 2회 실행):

```
core 배정 일치   96%  (46/48, 표본 2회)   ← 오분류율의 계산 단위
판정 일치        81%  (61/75)             ← status·role·비중근거 삼중조
재현율 잡음      ±4%p                     ← 이보다 작은 차이는 개선이 아니다
```

골든셋 리포트가 **재현 가능한 행과 아닌 행을 분리해** 출력한다.
이 값을 안 재던 시절, 택소노미를 확장한 직후 재현율이 80.3% → 73.8%로 내려간
것을 하마터면 택소노미 탓으로 결론 낼 뻔했다. 원인은 실행 간 흔들림이었다.

### 4. 촉매는 "공시된 사실"만

`"AI 수요 수혜"` 같은 전망은 촉매가 아니다. 대조할 원문이 없으면 환각률을
잴 수 없고, 그러면 위 세 가지가 전부 무너진다.

**공시 유형이 촉매 종류를 정하고, LLM은 개입하지 않는다.** 신뢰도는 데이터
조회 6가지의 통과 수(0~6 정수)다 — 모델이 매긴 소수점이 아니라서 재현된다.

| | 촉매 | 앵커 | 크기 출처 |
|---|---|---|---|
| C1 | 자사주 취득·소각 | `주요사항보고서(자기주식취득결정)` | DART 구조화 API |
| C2 | 대형 수주 | `단일판매ㆍ공급계약체결` | 공시 본문 `매출액 대비(%)` |
| C3 | 실적 서프라이즈 | `연결재무제표기준영업(잠정)` | 자체 `op_delta_q` / 시총 |
| **X1** | **지분 희석** | `유상증자결정`·`전환사채권발행결정` | DART 구조화 API |

**역촉매를 같은 파이프라인에서 뽑고 리포트 위쪽에 배치한다.** 촉매만 찾는
시스템은 낙관 편향을 갖는다 — 실측으로 주요사항보고의 **41%가 유상증자·CB**였다.

---

## 빠른 시작

### 준비

```bash
pip install -r requirements.txt
```

리포 루트에 `.env`:

```
DART_API_KEY=<40자리>          # https://opendart.fss.or.kr 무료 발급
ANTHROPIC_API_KEY=sk-ant-...   # LLM 태깅용. 없으면 --dry-run 까지만
```

### 처음부터 끝까지

```bash
python -m pipeline.cli init                                   # 스토어 생성
python -m pipeline.cli ingest-master --with-dart              # 상장사 마스터
python -m pipeline.cli ingest-dart --year 2025 --quarter 4    # 재무제표
python -m pipeline.cli ingest-prices --as-of 2026-08-06 --with-facts
python -m pipeline.cli derive --as-of 2026-08-06              # 파생지표 107개

python -m pipeline.cli screen  --as-of 2026-08-06 --screen deep_value --rebalance
python -m pipeline.cli enrich  --as-of 2026-08-06 --screen deep_value
python -m pipeline.cli tag     --as-of 2026-08-06 --screen deep_value   # 💰 LLM
python -m pipeline.cli verify  --as-of 2026-08-06 --screen deep_value
python -m pipeline.cli golden  --as-of 2026-08-06 --screen deep_value   # 오분류율
python -m pipeline.cli report  --as-of 2026-08-06 --screen deep_value
```

결과: `data/out/deep_value/2026-08-06/digest.md`

### 촉매까지

```bash
python -m pipeline.cli ingest-disclosures --lookback 180      # 공시 목록
python -m pipeline.cli catalysts --as-of 2026-08-06 --screen theme_hunt
python -m pipeline.cli report    --as-of 2026-08-06 --screen theme_hunt
```

### 일별 운영

```powershell
.\scripts\daily_prices.ps1     # 작업 스케줄러 16:10 등록. 등록법은 파일 주석 참조
```

시세 갱신 → 파생지표 → 등록된 전 스크린 스캔. **바스켓은 갱신하지 않는다** —
확정(리밸런스)은 사람이 판단해 주/월 단위로 따로 돌린다.

---

## 스크린 네 종

하나의 엔진(L0~L7)에 **L3 게이트 설정만 갈아끼운다.**

| 스크린 | 무엇을 찾나 | 게이트 | 통과 |
|---|---|---|---:|
| `deep_value` | 싼 것 | PER↓ PBR↓ FCFY↑ | 65 |
| `garp` | 잘 크는데 안 비싼 것 | ROE↑ 성장↑ **PER 10~25 고정** | 45 |
| `quality_fcf` | 현금 잘 버는 것 | FCFY↑ OPM↑ (밸류에이션 무관) | 44 |
| `theme_hunt` | 테마에 누가 있나 | 업종·규모·유동성만 | 450 |

**딥밸류와 GARP의 교집합은 0종목이다.** 싼 것만 보는 스크린은 한국카본
(ROE 16.4% / PER 12.5)이나 HD현대마린엔진(ROE 35.9% / PER 10.7)을 영영 보지
못한다. 스크린 다변화의 근거는 이 실측이다.

`theme_hunt`는 성격이 다르다 — 밸류에이션 게이트가 없어 **보유 바스켓이 아니라
조사 대상 목록**이다. 골든셋 라벨이 없어 오분류율을 재지 않는다.

---

## 명령어

| 명령 | 하는 일 | LLM |
|---|---|---|
| `init` | DuckDB 스토어·디렉터리 생성 | |
| `status` | 스토어 적재 현황 | |
| `ingest-master` | KIND 상장사 목록·업종·관리종목 | |
| `ingest-dart` | 재무제표 (종목당 1콜) | |
| `probe-price` | 가격 소스 소량 검증 — **대량 수집 전 필수** | |
| `ingest-prices` | 종가·주식수·시총 | |
| `ingest-dividend` / `ingest-holder` | 배당·최대주주 | |
| `ingest-disclosures` | 공시 목록 (촉매 앵커) | |
| `renormalize` | 캐시된 raw 로 facts 재생성 — **API 0회** | |
| `derive` | 파생지표 107개 + 품질 플래그 | |
| `checks` | 스크린 체크 목록·on/off·데이터 유무 | |
| `screen` | 적응형 게이트 → 생존 종목 | |
| `enrich` | 사업보고서 섹션 발췌 → evidence pack | |
| `tag` | 테마 분류 | 💰 |
| `verify` | V1 인용대조 / V2 섹터 / V3 비중 | |
| `catalysts` | 공시 → 촉매·역촉매 | |
| `golden` | 골든셋 대비 오분류율 | |
| `report` | 다이제스트 | |

`--screen {deep_value,garp,quality_fcf,theme_hunt}` — 산출물 경로도 함께 갈린다.

---

## 지금 어디까지 왔나

**작동한다**

- PIT(point-in-time) 스토어. 모든 조회가 `available_at <= as_of` — 룩어헤드 없음
- 정정공시는 UPDATE 하지 않고 append (`revision_of`)
- raw 불변 → `renormalize` 로 파서 버그를 **API 호출 0회**로 소급 수정
- 적응형 게이트: tightness 이분탐색으로 LLM 투입 종목 수 통제
- 프롬프트 캐싱 + `pack_hash` 캐시. `pack_hash`가 **주가와 무관**해서 일별
  시세 갱신에 LLM 비용이 0원이다
- 검증 레이어 V1~V3. 폐기 사유는 인용 환각과 사전 미등록 둘뿐
- 촉매·역촉매 1,188건, 신뢰도 6/6이 135건. **LLM 호출 0회**
- 테스트 398개

**한계 — 알고 있고 감추지 않는다**

- **재현성이 100%가 아니다.** core 96% / 판정 81%. Sonnet 5가 `temperature`를
  받지 않고(400 deprecated) 승급률이 85~92%라 대부분의 호출을 고정할 수 없다
- **골든셋은 사람 검수 전이다.** `labeled_by: assistant-bootstrap`
- **`theme_hunt`는 오분류율 미측정.** 450종목 라벨링을 안 했다
- **촉매의 `evidence_quote`가 아직 없다.** 지금 `chk_grounded`는 `rcept_no`
  존재로 대체 중 — 원문 링크는 있어 사람이 확인할 수 있다
- **상장폐지 이력이 없다.** KIND 현재 스냅샷만 있어 백테스트에 생존편향이 있다
- **`suspended`(거래정지) 가드는 소스가 없다.** 매 실행 경고한다
- **섹터 매핑이 지주사를 '기타 금융업'으로 보낸다.** 105종목이 `OTHER_FIN`이라
  제조 지주사(LS 등)가 재무 체크를 면제받는다

---

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 레이어별 작동 방식과 **왜 그렇게 했는지**(실측 근거) |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | 주기별 실행, 비용, 트러블슈팅 |
| [`tests/golden/LABELING.md`](tests/golden/LABELING.md) | 골든셋 라벨링 규칙 6가지 |
| [`docs/DESIGN.md`](docs/DESIGN.md) | 초기 설계서 — **일부 낡음**(등급 체계 시절) |

설정은 전부 YAML이고 코드 수정 없이 바꾼다:
`configs/screen/*.yaml` (게이트·체크) · `configs/themes/taxonomy_v1.yaml` (47테마)
· `configs/catalysts/catalyst_v1.yaml` (촉매 12종)

---

## 면책

이 저장소는 **리서치 아이디어 생성 도구**다. 매수·매도 신호가 아니며,
어떤 종목에 대한 투자 권유도 아니다. 데이터에는 알려진 결함이 있고(위 한계 참조)
LLM 산출물에는 오분류가 존재한다(측정치 3~6%). 투자 판단과 그 결과는
전적으로 사용자의 책임이다.
