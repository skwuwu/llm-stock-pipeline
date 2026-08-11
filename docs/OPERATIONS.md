# 운영 — 무엇을 언제 돌리나

주기가 셋이다. **섞으면 안 된다** — 일별 스캔이 바스켓을 갱신하면 히스테리시스
기준이 매일 어제로 밀려 바스켓이 서서히 표류한다.

| 주기 | 무엇 | 비용 | 왜 이 주기인가 |
|---|---|---|---|
| **일별** | 시세 → 파생 → 전 스크린 **스캔** | 0원 | 주가만 매일 바뀐다 |
| **주/월** | 바스켓 **리밸런스** → LLM 재태깅 | 💰 | 코호트를 바꾸면 골든셋 측정이 흔들린다 |
| **이벤트** | 공시 수집 → 촉매 갱신 | 0원 | 공시는 매일 나오지만 LLM을 안 쓴다 |

---

## 일별 — 시세 갱신

```powershell
.\scripts\daily_prices.ps1
```

작업 스케줄러 등록 (관리자 권한 불필요):

```powershell
$repo = "C:\Users\gimgy\OneDrive\바탕 화면\LLM_stock_pipeline"
$act  = New-ScheduledTaskAction -Execute "powershell.exe" `
          -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\daily_prices.ps1`""
$trg  = New-ScheduledTaskTrigger -Daily -At 16:10
Register-ScheduledTask -TaskName "LLM-stock daily prices" -Action $act -Trigger $trg
```

**16:10인 이유**: 정규장 마감 15:30 + 종가 확정·데이터 반영 여유.

**요일 조건을 걸지 않는다.** 스크립트가 마지막 거래일을 스스로 찾고 이미
받았으면 건너뛴다. 조건을 스케줄러와 스크립트 두 군데 두면 임시휴장 때 어긋난다.

### 이 스크립트가 지키는 것

**`as_of`는 오늘이 아니라 마지막 거래일이다.** 토요일에 오늘 날짜로 돌리면
금요일 종가가 토요일 행으로 들어가 휴장일에 유령 행이 생긴다.
거래일은 **공휴일 달력 대신 프로브로** 정한다 — 달력은 매년 갱신해야 하고
임시휴장을 못 잡는다.

**부분 적재를 완료로 오인하지 않는다.** 커버리지가 90% 미만이면 **비정상
종료**한다. 행이 있다는 것과 다 받았다는 것은 다르다 — 4종목만 들어간 상태를
'완료'로 읽으면 그날 시세가 영영 안 채워진다.

**멱등성은 단계별이다.** 수집을 건너뛴다고 `--derive`/`--scan`까지 건너뛰면,
수동으로 `ingest-prices`를 돌린 뒤 후속을 붙여도 아무 일도 안 난다.

### 로그

```
data/logs/daily_prices.jsonl     실행 기록 (JSONL)
data/logs/run_YYYY-MM-DD.txt     콘솔 출력
```

한글이 깨지면 `PYTHONIOENCODING`과 `[Console]::OutputEncoding` **둘 다**
설정됐는지 확인할 것. 하나만 하면 파일에 깨진 글자가 남는다.

---

## 주/월 — 리밸런스

바스켓을 확정하고 LLM 단계를 다시 돈다. **비용이 드는 유일한 주기다.**

```bash
D=2026-08-06
S=deep_value

python -m pipeline.cli screen  --as-of $D --screen $S --rebalance
python -m pipeline.cli enrich  --as-of $D --screen $S
python -m pipeline.cli tag     --as-of $D --screen $S      # 💰
python -m pipeline.cli verify  --as-of $D --screen $S
python -m pipeline.cli golden  --as-of $D --screen $S
python -m pipeline.cli report  --as-of $D --screen $S
```

### 리밸런스 전에 볼 것

```bash
python -m pipeline.cli screen --as-of $D --screen $S           # 스캔 (바스켓 불변)
```

출력에 진입·이탈 후보가 나온다. **표류가 작으면 리밸런스하지 않는 것이 맞다** —
코호트가 바뀌면 골든셋 라벨과 어긋나 오분류율이 "데이터 품질"이 아니라
"오늘의 종목 구성"을 재게 된다.

### 임계값 튜닝

```bash
python -m pipeline.cli checks --as-of $D --screen $S            # 체크 목록·데이터 유무
python -m pipeline.cli screen --as-of $D --screen $S --preview  # 적중 수·분포만
python -m pipeline.cli screen --as-of $D --screen $S --enable X --disable Y
```

`--preview`는 스크린하지 않고 체크별 적중 수만 낸다. 설정 파일을 고치기 전에
이걸로 확인한다.

> ⚠ 체크를 `gate_filter`로 바꿔도 **종목 수는 줄지 않는다.** 이분탐색이
> `target_count`를 맞추려고 `t`를 조정하므로 바뀌는 것은 개수가 아니라 구성이다.
> 개수를 줄이려면 `hard_guard`로 두거나 `target_count`를 낮출 것.

---

## 이벤트 — 공시·촉매

```bash
python -m pipeline.cli ingest-disclosures --lookback 180
python -m pipeline.cli catalysts --as-of 2026-08-06 --screen theme_hunt
python -m pipeline.cli report    --as-of 2026-08-06 --screen theme_hunt
```

**LLM을 쓰지 않으므로 자주 돌려도 된다.** 일별 스크립트에 붙여도 무방하다.

증분 수집은 `--lookback`을 짧게 준다. `rcept_no`가 PK라 재수집해도 중복되지
않는다.

```bash
python -m pipeline.cli ingest-disclosures --lookback 7    # 매일 돌릴 때
```

---

## 비용

### DART — 무료, 일 20,000콜

| 작업 | 콜 수 |
|---|---|
| 공시 목록 180일 (B+I) | 440 |
| 재무제표 | 종목당 1 |
| 배당·최대주주·주식수 | 종목당 각 1 |
| 촉매 구조화 API | (종목 × 엔드포인트) 수백 |
| 사업보고서 본문 | 종목당 1 |

`renormalize`는 **0콜**이다 — 캐시된 raw로 facts를 재생성한다.

### LLM — 실측

종목당 대략 **$0.02~0.03**. 승급률 85~92%(대부분 Sonnet 5까지 간다).

| 코호트 | 대략 |
|---|---|
| 40~65종목 (deep_value/garp/quality_fcf) | $1~2 |
| 450종목 (theme_hunt) | $8~10 |

**캐시가 스크린 간 공유**되므로 겹치는 종목은 두 번째부터 무료다.
`pack_hash`가 주가와 무관해서 **일별 시세 갱신에는 LLM 비용이 0원**이다.

`--dry-run`으로 비용만 추정할 수 있다:

```bash
python -m pipeline.cli tag --as-of $D --screen $S --dry-run
```

---

## 설정 바꾸기

전부 YAML이고 코드를 고치지 않는다.

### 스크린 (`configs/screen/*.yaml`)

```yaml
universe:
  exclude_flags: [...]        # 하드가드. 컬럼이 없으면 실행이 실패한다
  include_sectors: [...]      # 섹터 화이트리스트 (theme_hunt 전용)
  min_market_cap_krw: ...
filters:                      # tightness 로 보간되는 정량 필터
  - {metric: per, direction: lower_better, loose: 15.0, tight: 6.0}
gate:
  mode: target_count          # fixed | target_count | rank_top_n
  target_count: 60
checks:                       # 켜고 끄는 조건. kind 로 역할이 정해진다
  - id: low_roic
    kind: soft_flag           # hard_guard | gate_filter | soft_flag
    enabled: true
    metric: roic
    op: "<"
    threshold: 0.05
ranking:
  weights: {per: -0.40, pbr: -0.30, fcf_yield: 0.30}
```

새 스크린을 추가하려면 YAML을 만들고 `src/pipeline/screen/registry.py`의
`SCREENS`와 `GOLDEN_LABELS`에 등록한다. 라벨이 없으면 빈 목록으로 **명시**한다
— 테스트가 강제한다.

### 테마 (`configs/themes/taxonomy_v1.yaml`)

47테마. 각각 `inclusion` / `exclusion` / `allowed_sectors` /
`core_revenue_share_min`.

> ⚠ **테마를 추가하면 `include_sectors`도 확인할 것.** 실측: 의료기기가
> 5종목뿐이었는데 한국에 의료기기 회사가 없어서가 아니라 `HEALTHCARE_SVC`를
> 안 열어서였다. 넓히니 17종목이 됐다.

> ⚠ **택소노미를 고치면 LLM 캐시가 전부 무효화된다.** `system_fingerprint`가
> 프롬프트 내용 해시라서다. 등록된 전 스크린을 재태깅해야 한다.

### 촉매 (`configs/catalysts/catalyst_v1.yaml`)

12종(활성 4). `enabled: true`만 바꾸면 켜진다. 켜기 전에 패턴이 실물과 맞는지
확인할 것:

```bash
python -c "
import sys,yaml,re,duckdb; sys.path.insert(0,'src')
from pathlib import Path
cat=yaml.safe_load(Path('configs/catalysts/catalyst_v1.yaml').read_text(encoding='utf-8'))
d=duckdb.connect('data/pit.duckdb',read_only=True).execute(
    'SELECT * FROM disclosures WHERE ticker IS NOT NULL').df()
for c in cat['catalysts']:
    p=re.compile('|'.join(c['patterns']))
    n=len(d[d.pblntf_ty.isin(c['pblntf_ty']) & d.report_key.str.contains(p,regex=True)])
    print(f\"  {c['id']:<3}{c['name']:<14}{n:>6}건  enabled={c['enabled']}\")
"
```

---

## 트러블슈팅

### `MissingGuardError: 하드 가드 컬럼 없음`

`derive`를 안 돌렸거나 체크가 요구하는 지표가 없다. **조용히 건너뛰지 않는
것이 의도된 동작이다** — 가드가 도는 줄 알고 안 도는 상태가 더 위험하다.

### `중단: 측정 대상 N종목 중 X% 만 라벨이 있다`

바스켓이 바뀌었는데 골든셋을 갱신하지 않았다. 라벨을 추가하거나
`--allow-cohort-drift`로 감수한다(값의 의미가 달라진다는 뜻).

### `스크린 'X' 의 골든셋 라벨이 없다`

의도된 거절이다. 라벨 없이 0%를 내면 그 숫자를 신뢰하게 된다.
`registry.GOLDEN_LABELS`에 등록하거나 `--golden <경로>`로 지정한다.

### 시세 결측이 갑자기 늘었다

`tests/test_m5_share_class.py::test_market_cap_coverage_regression`이 5% 초과
시 실패한다. DART `stockTotqySttus`의 `se` 라벨에 새 표기가 등장했을 수 있다:

```bash
python -c "
import sys,json,collections; sys.path.insert(0,'src')
from pathlib import Path
from pipeline.ingest.prices import classify_share_class as cls
unk=collections.Counter()
for f in Path('data/raw/dart/stockTotqySttus').glob('*.json'):
    p=json.loads(f.read_text(encoding='utf-8'))
    if p.get('status')!='000': continue
    for r in p.get('list',[]):
        se=' '.join((r.get('se') or '').split())
        if se and '합계' not in se and se!='비고' and cls(se) is None: unk[se]+=1
print(unk.most_common(20))
"
```

### 지표가 갑자기 전부 NaN

`assert_metrics_sane`이 `derive`에서 잡는다. `_safe_div`의 스칼라 분모
브로드캐스트 같은 고장은 예외를 던지지 않아 테스트가 늘어도 안 잡힌다 —
산출물 자체를 검사해야 한다.

### 재실행했는데 결과가 안 바뀐다

캐시 키를 확인한다. 프롬프트·택소노미·온도를 바꿨으면
`system_fingerprint`가 달라져야 한다:

```bash
python -c "
import sys,yaml; sys.path.insert(0,'src')
from pathlib import Path
from pipeline.llm.cascade import system_fingerprint
tx=yaml.safe_load(Path('configs/themes/taxonomy_v1.yaml').read_text(encoding='utf-8'))
print('현재:', system_fingerprint(tx))
import json
print('태그 파일:', json.loads(Path('data/llm/deep_value/tags_2026-08-06.json')
      .read_text(encoding='utf-8'))[0].get('system_fingerprint'))
"
```

### 지표 변화가 개선인지 잡음인지 모르겠다

재현성을 재라. **재현율 ±4%p보다 작은 차이는 잡음이다.**

```bash
python scripts/determinism_probe.py --screen deep_value --n 24
```

캐시를 우회해 같은 pack을 두 번 태깅하고 core/판정/전체 일치율을 낸다.
결과가 나오면 `src/pipeline/verify/golden.py`의 `_SAMPLES`에 **누적**한다 —
단일 실행 수치는 양쪽으로 튄다.

---

## 데이터 배치

```
data/
  pit.duckdb                       PIT 스토어 (facts·prices·master·disclosures)
  raw/**                           불변. 절대 덮어쓰지 않는다
  derived/metrics_{date}.parquet   전 종목 파생지표 — 스크린을 타지 않는다
  llm/tags/                        pack_hash 캐시 — 스크린 간 공유
  screens/{screen}/{date}/         survivors · why_excluded · manifest · catalysts
  screens/{screen}/_basket.json    확정 바스켓 (리밸런스에서만 갱신)
  enrich/{screen}/{date}/          evidence pack
  llm/{screen}/tags_{date}.json    태깅 결과
  verify/{screen}/{date}/          verdicts · metrics · golden_metrics
  out/{screen}/{date}/digest.md    다이제스트
```

**`derived/`와 `llm/tags/`는 스크린별로 나누지 않는다.** 전자는 전 종목
대상이라 스크린을 안 타고, 후자는 `pack_hash` 키라 공유해야 이득이다.

---

## 테스트

```bash
python -m pytest tests/ -q
```

398개. 산출물이 없으면 `skip` 되므로 클린 체크아웃에서도 돈다.

회귀 방지 성격의 테스트가 많다 — 이름을 보면 어떤 버그를 막는지 알 수 있다:

```
test_m2_integrity.py::test_no_orphan_derived_metrics       계산하고 안 읽는 지표
test_m3_multiscreen.py::test_cli_does_not_hardcode_a_screen_config
test_m4_reproducibility.py::test_reproducibility_is_pooled_not_a_single_run
test_m5_share_class.py::test_market_cap_coverage_regression
test_m6_catalysts.py::test_confidence_checks_are_all_deterministic
```
