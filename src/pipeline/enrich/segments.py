"""사업보고서 본문 → 구조화된 세그먼트 매출 비중.

V3 검증이 실제로 동작하려면 '실제 비중'이 숫자로 있어야 한다. 그게 없으면
LLM 이 주장한 비중을 테마 하한과 비교하는 것뿐이고, 그건 LLM 을 LLM 으로
검증하는 셈이라 아무것도 못 잡는다.

파싱 실패를 조용히 넘기지 않는다. 그리고 **입증 가능성을 등급으로 남긴다**:

  corroborated=True
    prose         회사가 문장에 직접 % 를 밝힌 값
    table_amount  금액 합이 DART 매출액과 일치함을 확인한 값
  corroborated=False
    table_pct     비율 컬럼. 독립적으로 대조할 수단이 없다.

V3 는 corroborated 인 수치로만 강등한다. 뒷받침 없는 수치로 배정을 깎으면
검증이 아니라 또 하나의 추측이 된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "DX 부문이 174조 8,877억원(58.1%)" / "SDC가 29조 1,578억원(9.7%)"
_AMOUNT_SHARE = re.compile(
    r"([가-힣A-Za-z0-9·&\-\s]{1,24}?)\s*(?:부문|사업부문|사업부|본부)?\s*(?:이|가|은|는)?\s*"
    r"((?:[\d,]+조\s*)?[\d,]+\s*(?:억원|백만원|천원|원))\s*\(\s*([\d.]+)\s*%\s*\)"
)
# "중전기기 부문 매출 비중 72%" / "반도체부문 72.0%"
#   / "기계사업부문은 연결매출의 83.6%" ← 부문과 숫자 사이 연결어구를 허용해야 한다.
#     (실측: 화천기공. LLM 이 이 문장을 인용했는데 파서는 못 읽고 있었다)
_SHARE_ONLY = re.compile(
    r"([가-힣A-Za-z0-9·&\-]{1,20}?)\s*(?:사업부문|사업부|부문)\s*"
    r"(?:은|는|이|가)?\s*(?:[가-힣]{0,6}(?:매출액?|비중)의?\s*)?"
    r"(?:매출\s*)?(?:비중\s*)?(?:약\s*)?([\d.]+)\s*%"
)

_NOISE = ("합계", "총계", "소계", "전체", "기타부문", "연결조정", "내부거래", "조정")
# 증감률·성장률 문장은 매출 '구성'이 아니다.
# 실측 오탐: "313백만원으로 전년동기대비 4%" 를 세그먼트 4% 로 읽었다.
_RATE_CONTEXT = ("전년동기", "전년 동기", "전년대비", "전년 대비", "증감", "성장률",
                 "감소하", "증가하", "대비 ", "yoy", "YoY")
SUM_TOLERANCE = 0.12          # 서술문 비중 합 허용 오차
# 비율 컬럼은 교차검증 수단이 없으므로 좁게 잡는다.
# 넓게 두면 합 109%/112% 짜리 다른 표(해외사업·점유율)가 매출 구성으로 통과한다.
PCT_SUM_TOLERANCE = 5.0
MIN_SEGMENTS = 2


@dataclass
class Segment:
    name: str
    share: float                  # 0..1
    amount_text: str | None = None
    # rowspan 표의 라벨 경로 전체("제품/금형/용역 국내법인 CANISTER").
    # 세그먼트로서 의미 있는 이름이 사업부문 열이 아니라 품목 열에 있는 표가 흔해서
    # (실측: 코리아에프티는 CANISTER·FILLER NECK 이 품목 열에 있다),
    # 표시용 이름과 조회용 경로를 나눠 둔다.
    path: str = ""


@dataclass
class SegmentSet:
    segments: list[Segment] = field(default_factory=list)
    complete: bool = False        # 비중 합이 1.0 근처인가
    source_line: str = ""
    method: str = ""              # prose | table_pct | table_amount — 감사용
    context_score: int = 0        # 문맥이 매출 구성표임을 얼마나 강하게 말하는가
    corroborated: bool = False    # 독립적으로 뒷받침되는 수치인가 (아래 설명)

    def share_of(self, name: str) -> float | None:
        """세그먼트명으로 비중 조회. 부분일치·공백무시로 관대하게 찾는다.

        LLM 이 '중전기기'라고 쓰고 원문이 '중전기기부문'인 경우를 살린다.
        표에서 온 이름은 rowspan 라벨 경로 전체("타이어 제품 타이어 한국")라
        부분일치가 사실상 필수다.

        **여럿에 걸리면 None.** 어느 것인지 모르는데 하나를 골라 돌려주면
        V3 가 엉뚱한 비중으로 강등한다. 모호함은 확인 불가로 처리해야 한다.
        """
        if not name:
            return None
        key = _norm(name)
        if len(key) < 2:
            return None
        for s in self.segments:
            if _norm(s.name) == key:
                return s.share

        def hit(s: Segment) -> bool:
            for field_ in (s.name, s.path):
                v = _norm(field_)
                if len(v) >= 2 and (v in key or key in v):
                    return True
            return False

        hits = [s.share for s in self.segments if hit(s)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1 and max(hits) - min(hits) < 1e-9:
            return hits[0]              # 같은 값이면 모호하지 않다
        return None

    def as_dict(self) -> dict:
        return {"complete": self.complete, "source_line": self.source_line,
                "method": self.method, "corroborated": self.corroborated,
                "segments": [{"name": s.name, "share": round(s.share, 4),
                              "amount": s.amount_text, "path": s.path}
                             for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "SegmentSet":
        if not d:
            return cls()
        return cls(segments=[Segment(s["name"], float(s["share"]), s.get("amount"),
                                     s.get("path", ""))
                             for s in d.get("segments", [])],
                   complete=bool(d.get("complete")),
                   source_line=d.get("source_line", ""),
                   method=d.get("method", ""),
                   corroborated=bool(d.get("corroborated")))


def _norm(s: str) -> str:
    return re.sub(r"[\s·\-&()]", "", s or "").lower()


def _clean_name(raw: str) -> str:
    n = raw.strip(" ,·")
    # 앞에 붙은 서술 조각을 잘라낸다: "2024년 매출은 DX" → "DX"
    n = re.split(r"매출은|매출액은|매출이|입니다|하였습니다|,", n)[-1]
    n = re.sub(r"^\d{4}년\s*", "", n).strip()
    return n[-20:].strip()


def parse_segments(text: str, revenue: float | None = None) -> SegmentSet:
    """본문(서술문 + 표)에서 세그먼트 비중을 뽑는다.

    후보가 여럿이면 '세그먼트 수가 많고 합이 1.0 에 가까운' 것을 고른다.
    서술문과 표가 모두 있으면 더 완전한 쪽이 이긴다.
    """
    best: SegmentSet | None = None
    for cand in _prose_candidates(text):
        # 서술문은 항상 corroborated 라 단일 추출을 걸러낼 근거가 없다.
        # 문장 하나에서 % 하나만 잡힌 것은 대개 오독이다
        # (실측: 한전 '기타' 100%, 폴라리스AI파마 '내수와 수출' 100%).
        if len(cand.segments) < MIN_SEGMENTS or _is_degenerate(cand):
            continue
        if best is None or _rank(cand) > _rank(best):
            best = cand
    for cand in _table_candidates(text, revenue):
        # 표의 단일 사업부문은 매출 대조를 통과했을 때만 인정한다(_parse_table 참조).
        if not cand.segments or _is_degenerate(cand):
            continue
        if best is None or _rank(cand) > _rank(best):
            best = cand
    return best or SegmentSet()


def _prose_candidates(text: str):
    for line in _candidate_lines(text):
        yield _parse_line(line)


# ── 표 파싱 ─────────────────────────────────────────────────────────
# 대부분의 중소형주는 매출 구성을 서술문이 아니라 **표**로 보고한다.
# 비율 컬럼이 아예 없고 금액만 있는 경우가 흔해(실측: 성신양회) 정규화가 필요하다.
_TABLE_KEYS = ("사업부문", "부문", "품목", "매출유형", "제품", "구분", "사업")
_NUM_CELL = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$")
MAX_TABLE_ROWS = 40


# 표의 첫 행이 곧 헤더인 것은 아니다. DART 는 '(단위 : 백만원)' 같은 단위 표기를
# 별도 행으로 먼저 넣는 경우가 매우 흔하고, 그러면 첫 행에는 열 이름이 하나도 없다.
# 첫 행만 보고 판정하면 정상 매출실적표가 통째로 버려진다
# (실측: 팜스토리·넥센타이어·KG스틸 모두 이 이유로 세그먼트 0건이었다).
_HEADER_SCAN_ROWS = 3


def _header_text(block: list[list[str]]) -> str:
    return " ".join(" ".join(r) for r in block[:_HEADER_SCAN_ROWS])


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.split("|") if c.strip()]


def _num(cell: str) -> float | None:
    t = cell.replace(",", "").replace("%", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def _table_blocks(text: str) -> list[list[list[str]]]:
    """연속된 표 행 묶음. 각 묶음은 셀 리스트의 리스트.

    행 사이의 **빈 줄은 무시한다** — 평문화 과정에서 행마다 빈 줄이 끼어들어,
    빈 줄에서 블록을 끊으면 표가 한 행짜리로 쪼개진다(실측: 블록 0개였다).
    """
    lines = text.splitlines()
    return [b for _, b in _table_blocks_with_pos(lines, [l.count("|") >= 2 for l in lines])]


def _table_blocks_with_pos(lines: list[str], is_row: list[bool]):
    """(시작 줄번호, 블록) 목록. 문맥 판정을 위해 위치를 함께 돌려준다."""
    out, cur, start = [], [], 0
    for i, line in enumerate(lines):
        if is_row[i]:
            if not cur:
                start = i
            cur.append(_cells(line))
        elif line.strip():                 # 내용 있는 비표 줄에서만 끊는다
            if len(cur) >= 3:
                out.append((start, cur[:MAX_TABLE_ROWS]))
            cur = []
    if len(cur) >= 3:
        out.append((start, cur[:MAX_TABLE_ROWS]))
    return out


# 매출 구성표가 아닌 표를 걸러낸다. 실측 오탐:
#   샘표식품 건물/구축물/기계장치(유형자산), 팜스토리 도축두수/가동률(생산능력),
#   네오위즈 등록/출원(특허), KSS해운 매출채권/금융자산(금융상품),
#   고려산업 금융수익/감가상각비(부문별 손익)
_NON_REVENUE_NAMES = (
    "토지", "건물", "구축물", "기계장치", "차량운반구", "공구", "비품", "건설중인",
    "가동률", "생산능력", "생산실적", "도축", "가동시간", "가동일수",
    "금융자산", "금융부채", "금융수익", "금융비용", "매출채권", "수취채권",
    "금융상품", "감가상각", "무형자산상각", "법인세", "영업이익", "당기순이익",
    # 손익계산서 항목. 사업부문이 아니라 같은 매출을 쪼갠 것이다
    # (실측: HS애드가 '매출액 67% / 매출총이익 33%' 로 나왔다).
    "매출액", "매출총이익", "매출원가", "판매비", "관리비",
    "부문자산", "부문부채", "등록", "출원", "특허", "종업원", "임직원",
)
_REVENUE_CONTEXT = ("매출", "판매", "수주")
# 매출과 무관한 비율표. 합이 정확히 100이라 비율 검증만으로는 걸러지지 않는다.
# 실측 오탐: 티에이치엔의 경쟁사 시장점유율 표(자사 13% + 경쟁사 46% + 41%).
_NEGATIVE_CONTEXT = ("점유율", "시장규모", "경쟁", "산업의 특성", "M/S", "시장 현황",
                     # 생산량·설비 표는 단위가 톤·대·㎡ 라 금액이 아니다. 그런데
                     # 배율(1e3/1e6/1e8) 중 하나가 매출과 우연히 맞을 수 있어
                     # 매출 대조만으로는 안 걸린다(실측: 팜스토리 사료 1,598,668톤
                     # × 1e6 이 매출 1.45조와 10% 차이로 통과했다).
                     "생산실적", "생산능력", "가동률", "생산설비", "재고현황",
                     # 원재료 매입은 매출의 반대편이다
                     "원재료", "매입실적", "매입액")
# 매출 구성표임을 강하게 시사하는 표현. 후보가 여럿일 때 순위를 가른다 —
# 실측: 문맥 창을 넓히자 E1(LPG)이 엉뚱한 표에서 '금융업 88%' 를 집어왔다.
_STRONG_CONTEXT = ("매출실적", "매출 실적", "매출 및 수주", "매출액 및 비중",
                   "매출구성", "매출 구성", "부문별 매출", "제품별 매출", "품목별 매출")
_CONTEXT_CHARS = 300
# "4. 매출 및 수주상황", "가. 매출실적" 같은 소제목. 표가 어느 절 아래에 있는지가
# 그 표가 매출 구성표인지를 가장 잘 말해준다.
_HEADING = re.compile(r"^\s*(?:\d{1,2}[.)]|[가-하][.)]|\([가-하\d]\))\s*\S")
_MAX_LOOKBACK_LINES = 80


def _heading_context(lines: list[str], is_row: list[bool], start: int) -> str:
    """표 위쪽으로 **가장 가까운 소제목까지** 거슬러 올라가 문맥을 모은다.

    고정 길이(300자)로 자르면 표 앞에 다른 표·주석이 끼어 있을 때 소제목에 못 닿는다
    (실측: 실패 46건 중 28건이 '매출 문맥 없음'이었는데 대부분 이 경우였다).
    """
    ctx: list[str] = []
    seen = 0
    for i in range(start - 1, max(-1, start - _MAX_LOOKBACK_LINES) - 1, -1):
        if is_row[i]:
            continue
        line = lines[i].strip()
        if not line:
            continue
        ctx.append(line)
        seen += len(line)
        if _HEADING.match(line):
            break                       # 소제목에 닿았다
        if seen > _CONTEXT_CHARS * 4:   # 소제목이 없으면 과도하게 올라가지 않는다
            break
    return " ".join(reversed(ctx))


def _table_candidates(text: str, revenue: float | None = None):
    """표 앞의 서술 문맥이 '매출'을 말하고 있을 때만 매출 구성표로 본다.

    헤더만 보고 고르면 유형자산·생산능력·특허 표를 매출 구성으로 읽는다
    (실측: 27건 중 절반이 오탐이었다). 헤더에 '매출'이 없는 정상 표도 흔해서
    (성신양회 헤더는 '사업부문|제품|구체적용도|제60기...'), 헤더 단독 판정은 불가능하다.
    """
    lines = text.splitlines()
    is_row = [l.count("|") >= 2 for l in lines]
    for start, block in _table_blocks_with_pos(lines, is_row):
        header = _header_text(block)
        if not any(k in header for k in _TABLE_KEYS):
            continue
        ctx = _heading_context(lines, is_row, start)
        if not any(k in ctx + header for k in _REVENUE_CONTEXT):
            continue
        if any(k in ctx + header for k in _NEGATIVE_CONTEXT):
            continue
        score = sum(1 for k in _STRONG_CONTEXT if k in ctx + header)
        cand = _parse_table(block, revenue, allow_unverified=score > 0)
        if cand and not _looks_non_revenue(cand):
            cand.context_score = score
            yield cand


def _looks_non_revenue(ss: SegmentSet) -> bool:
    bad = sum(1 for s in ss.segments if any(k in s.name for k in _NON_REVENUE_NAMES))
    return bad * 3 >= len(ss.segments)     # 1/3 이상이면 매출표가 아니다


# 사업 구분이 아니라 지역·법인 구분. 세그먼트로 쓰면 테마 검증에 아무 쓸모가 없고,
# 이런 이름이 그룹명으로 올라왔다는 건 rowspan 접기가 어긋났다는 신호이기도 하다.
_NOT_A_SEGMENT = {"내수", "수출", "국내", "해외", "지배회사", "종속회사",
                  "본사", "지점", "제품", "상품", "용역",
                  # 매출유형. 그룹명으로 올라왔다면 총합계 없는 표 뒤에 다른 표가
                  # 이어붙어 그 행을 사업부문으로 읽은 것이다(실측: 팜스토리).
                  "제품매출", "상품매출", "용역매출", "기타매출", "내부매출제거"}


# 지역·법인·매출유형 이름이 이만큼을 차지하면 그게 세그먼테이션 축이라는 뜻이다.
# 소액 잔여 항목으로 섞여 있는 것과는 구분해야 한다 — '전력기기 64% / 상품 11% /
# 신재생 24%' 에서 '상품'을 이유로 표 전체를 버리면 정상 파싱을 잃는다(실측: 피에스텍).
NOT_A_SEGMENT_SHARE_LIMIT = 0.40
# 사실상 단일 사업 회사는 실재한다(디바이스 오염제어 99.7%, 우원개발 토공 99.8%).
# 집계 실패와 구분하려면 문턱을 실제 관측치보다 위에 둬야 한다.
SINGLE_SEGMENT_LIMIT = 0.999


def _is_degenerate(ss: SegmentSet) -> bool:
    """구조적으로 망가진 파싱인가. **틀린 세그먼트는 없는 것보다 나쁘다** —
    V3 가 이 수치로 정상 배정을 강등하기 때문에 의심스러우면 버린다.

    다만 과하게 버리면 쓸 수 있는 수치까지 잃는다. 각 규칙은 '이 표가 사업부문
    구성표가 아니다'를 말할 수 있을 때만 발동해야 한다.
    """
    names = [_norm(s.name) for s in ss.segments]
    if len(set(names)) < len(names):
        return True          # 같은 그룹명 반복 = 행 묶기 실패 (실측: 한전 수주표)
    bad_share = sum(s.share for s, n in zip(ss.segments, names)
                    if n in _NOT_A_SEGMENT)
    if bad_share >= NOT_A_SEGMENT_SHARE_LIMIT:
        return True          # 지역·법인·매출유형이 세그먼테이션 축이다
    if len(ss.segments) >= 2 and max(s.share for s in ss.segments) >= SINGLE_SEGMENT_LIMIT:
        return True          # 하나가 100%, 나머지가 0 = 값이 엉뚱한 그룹으로 갔다
    if len(ss.segments) == 1 and (names[0] in _NOT_A_SEGMENT
                                  or any(k in ss.segments[0].name for k in _NOISE)):
        return True          # 단일인데 이름이 '기타'·'내수' 면 사업부문이 아니다
    return False


# 그룹 소계를 표시하는 셀. 이 행의 값이 그 사업부문의 매출이다.
_TOTAL_CELLS = ("소계", "합계", "총계", "계")


def _split_row(cells: list[str]) -> tuple[list[str], list[float]]:
    """행을 (라벨들, 기간별 숫자들)로 가른다. 숫자는 **오른쪽 끝에서부터** 잡는다.

    DART 매출실적표는 거의 전부 rowspan 표라서 행마다 셀 수가 다르다.
    사업부문·매출유형 셀이 rowspan 으로 묶여 있으면 이어지는 행에는 그 셀이
    아예 없다(빈 셀이 아니라 없다). 그래서 왼쪽 기준 열 정렬은 성립하지 않는다.
    기간 컬럼은 항상 오른쪽 끝에 붙으므로 그쪽에서부터 세는 것만이 안정적이다.
    """
    nums: list[float] = []
    i = len(cells)
    while i > 0:
        v = _num(cells[i - 1])
        if v is None:
            break
        nums.insert(0, v)
        i -= 1
    return cells[:i], nums


def _row_groups(block: list[list[str]]) -> list[tuple[str, str, float, bool]]:
    """rowspan 표를 (그룹명, 당기금액, 소계로_명시됨) 목록으로 접는다.

    그룹 시작 판정: 라벨 수가 최대인 행, 또는 소계 행 **직후**의 라벨 수가
    최대-1 인 행. 후자가 필요한 이유는 두 번째 사업부문부터는 상위 셀 하나가
    앞 그룹과 병합돼 라벨이 하나 적게 나오기 때문이다(실측: 넥센타이어 '기타').
    """
    parsed = [_split_row(c) for c in block if c]
    parsed = [(l, n) for l, n in parsed if n and l]
    if not parsed:
        return []
    width = max(len(n) for _, n in parsed)
    parsed = [(l, n[-width:]) for l, n in parsed if len(n) >= width]
    if not parsed:
        return []
    max_labels = max(len(l) for l, _ in parsed)

    groups: list[dict] = []
    cur: dict | None = None
    after_total = False
    for labels, nums in parsed:
        flat = [re.sub(r"\s+", "", x) for x in labels]
        if any(x in _TOTAL_CELLS for x in flat):
            if cur is not None:
                # 그룹 소계인가, 표 전체의 총합계인가. 값이 지금까지 누적한 전체와
                # 같으면 총합계다. 이걸 그룹 소계로 읽으면 마지막 사업부문에
                # 회사 전체 매출이 들어가 비중이 통째로 망가진다
                # (실측: 성신양회 기타 1.6% → 100%).
                # 그룹이 2개 미만이면 총합계일 수 없다 — 첫 사업부문의 소계는
                # 정의상 '지금까지 누적'과 같아서, 이 검사가 그걸 먹어버렸다.
                running = sum(g["value"] for g in groups)
                if len(groups) >= 2 and running > 0 \
                        and abs(nums[0] - running) <= running * 0.01:
                    # 총합계를 만나면 표는 끝이다. 평문화 과정에서 빈 줄만으로
                    # 구분된 다음 표가 같은 블록에 붙는 일이 흔한데, 여기서 멈추지
                    # 않으면 그 표의 행들이 사업부문으로 딸려 들어온다
                    # (실측: 팜스토리 합계가 1.66조 → 3.11조로 부풀었다).
                    break
                # 한 그룹 안에 소계가 여러 번 나오면(수출 소계·내수 소계·총계)
                # **첫 소계만** 쓴다. 덮어쓰면 마지막 작은 소계가 사업부문 매출로
                # 남는다(실측: 코메론 줄자사업 664억 → 14억).
                if not cur["explicit"]:
                    cur["value"], cur["explicit"] = nums[0], True
                after_total = True
            continue
        # 소계 뒤에 오는 비-소계 행은 반드시 새 사업부문이다. 라벨 수로만 판정하면
        # 상위 셀이 더 병합된 두 번째 사업부문을 연속행으로 오인한다
        # (실측: KG스틸 'PEB 등|내수' 가 철강부문에 흡수돼 그룹이 1개가 됐다).
        starts = len(labels) == max_labels or after_total
        if cur is None or starts:
            # 표시용 이름은 첫 라벨(대개 사업부문), 조회용 경로는 라벨 전체.
            # DART 는 '시 멘 트' 처럼 자간을 벌려 쓰므로 라벨 내부 공백은 없앤다.
            cur = {"name": flat[0][:24], "path": " ".join(flat)[:60],
                   "value": 0.0, "explicit": False}
            groups.append(cur)
        if not cur["explicit"]:
            cur["value"] += nums[0]
        after_total = False
    return [(g["name"], g["path"], g["value"], g["explicit"]) for g in groups
            if g["value"] > 0 and not _bad_name(g["name"])]


def _parse_table(block: list[list[str]], revenue: float | None = None,
                 allow_unverified: bool = False) -> SegmentSet | None:
    """표 → 세그먼트.

    비율 컬럼이 있으면 그걸 쓰고, 없으면 금액을 정규화한다.
    어느 컬럼을 썼는지 자기검증한다: 합이 100 근처면 비율 컬럼으로 본다.

    allow_unverified: 매출 대조에 실패해도 버리지 않고 corroborated=False 로 남긴다.
        **매출실적 절 아래에 있는 표에만** 허용해야 한다. 아무 표에나 허용하면
        단가표·거래처표가 다시 세그먼트로 들어온다 — 매출 대조는 바로 그걸 막으려고
        넣은 검사다.
    """
    # 1) 비율 컬럼. 열 정렬이 성립하는 단순 표에서만 의미가 있다.
    simple: list[tuple[str, list[float]]] = []
    for cells in block[1:]:
        if not cells:
            continue
        name = re.sub(r"\s+", "", cells[0])
        if _bad_name(name) or _NUM_CELL.match(cells[0].strip()) or len(name) > 20:
            continue
        nums = [v for v in (_num(c) for c in cells[1:]) if v is not None]
        if nums:
            simple.append((name, nums))
    if len(simple) >= MIN_SEGMENTS:
        arity = min(len(n) for _, n in simple)
        for j in range(arity):
            agg: dict[str, float] = {}
            ok = True
            for name, nums in simple:
                if nums[j] < 0:
                    ok = False
                    break
                agg[name] = agg.get(name, 0.0) + nums[j]
            if ok and len(agg) >= MIN_SEGMENTS and \
                    abs(sum(agg.values()) - 100.0) <= PCT_SUM_TOLERANCE:
                return SegmentSet(
                    segments=[Segment(k, v / 100) for k, v in agg.items()],
                    complete=True, method="table_pct",
                    source_line=" | ".join(block[0])[:200] + f"  (열 {j + 1}, table_pct)")

    # 2) 금액. rowspan 을 접어 사업부문 단위로 모은다.
    #    기간은 **당기(첫 숫자열)만** 본다. 당기가 검증에 실패했다고 전기로 넘어가면
    #    작년 매출 구성을 조용히 쓰게 된다.
    groups = _row_groups(block)
    if not groups:
        return None
    total = sum(v for _, _, v, _ in groups)
    if total <= 0:
        return None
    segs = [Segment(nm, v / total, f"{v:,.0f}", path) for nm, path, v, _ in groups]

    # **자기검증**: 매출 구성표라면 합이 그 회사 매출액과 맞아야 한다.
    # 단가표(원/톤), 거래처별 표를 어휘 목록 없이 걸러낸다.
    matched = revenue is not None and _matches_revenue(total, revenue)
    # 사업부문이 하나뿐인 회사는 실재하고, V3 에게는 그게 가장 쓸모 있는 정보다
    # ("이 회사는 사실상 100% 이 사업" 이면 core 배정을 그대로 확인해준다).
    # 다만 그룹이 하나면 '접기가 실패해 하나로 뭉친 것'과 구별할 수단이 매출 대조뿐이라,
    # **대조를 통과할 때만** 인정한다.
    if len(groups) < MIN_SEGMENTS and not matched:
        return None
    if revenue is not None and not matched:
        if not allow_unverified:
            return None
        # 파싱 성공과 뒷받침 가능을 구분한다. 매출실적표에는 '(내부거래포함)' 처럼
        # 연결매출과 정의가 다른 것이 흔해, 대조 실패를 곧바로 '매출표가 아니다'로
        # 읽으면 과잉 차단이 된다. 남기되 corroborated=False 로 V3 강등을 막는다.
        return SegmentSet(
            segments=segs, complete=True, method="table_amount_unverified",
            corroborated=False,
            source_line=" | ".join(block[0])[:200]
                        + f"  (열 1, 합={total:,.0f} 매출대조 실패)")
    return SegmentSet(segments=segs, complete=True, method="table_amount",
                      corroborated=matched,
                      source_line=" | ".join(block[0])[:200] + "  (열 1, table_amount)")


# 표 단위는 원/천원/백만원/억원이 섞여 있고 명시되지 않을 때도 있다.
# 배율을 맞춰보고 하나라도 매출과 맞으면 매출표로 인정한다.
_UNIT_SCALES = (1, 1_000, 1_000_000, 100_000_000)
# 배율 후보가 1000배씩 벌어져 있어도, 허용 오차가 크면 인접 배율이 우연히 맞을 수 있다
# (실측: 1,216,265 × 1e8 이 99조와 23% 차이로 통과했다).
REVENUE_MATCH_TOLERANCE = 0.20    # TTM 과 사업연도 기준 차이는 감안하되 좁게


def _matches_revenue(total: float, revenue: float) -> bool:
    if revenue <= 0 or total <= 0:
        return False
    return any(abs(total * s - revenue) / revenue <= REVENUE_MATCH_TOLERANCE
               for s in _UNIT_SCALES)


def _candidate_lines(text: str) -> list[str]:
    out = []
    for chunk in re.split(r"[\n]", text):
        t = chunk.strip()
        if not t or len(t) > 1200:
            continue
        if "%" not in t:
            continue
        if not any(k in t for k in ("매출", "부문", "사업부", "비중")):
            continue
        out.append(t)
    return out


def _is_rate_context(line: str, start: int, end: int) -> bool:
    """이 매칭이 증감률 서술인가. 줄 전체가 아니라 **매칭 주변만** 본다.

    줄 단위로 거르면 '시멘트 54%, 레미콘 12% ... 전년 대비 증가' 같은 정상 구성비
    문장까지 통째로 날아간다(실측: 성신양회가 그렇게 사라졌다).
    """
    left = line[max(0, start - 40): start]
    # 창 끝이 마침표로 끝나면(뒤에 공백이 안 잡힌 경우) 그 앞은 다른 문장이다.
    left = re.split(r"[.。](?:\s|$)", left)[-1]
    return any(k in left + line[start:end + 12] for k in _RATE_CONTEXT)


def _bad_name(name: str) -> bool:
    # '313백만원으로' 같은 금액 조각이 세그먼트명으로 잡히는 것을 막는다
    return (not name or any(x in name for x in _NOISE)
            or bool(re.search(r"\d\s*(?:원|억|조|백만|천만)", name)))


def _parse_line(line: str) -> SegmentSet:
    found: dict[str, Segment] = {}
    for m in _AMOUNT_SHARE.finditer(line):
        name, amount, pct = _clean_name(m.group(1)), m.group(2), float(m.group(3))
        if _bad_name(name) or not 0 < pct <= 100 or _is_rate_context(line, m.start(), m.end()):
            continue
        found.setdefault(_norm(name), Segment(name, pct / 100, amount))
    if len(found) < MIN_SEGMENTS:
        for m in _SHARE_ONLY.finditer(line):
            name, pct = _clean_name(m.group(1)), float(m.group(2))
            if _bad_name(name) or not 0 < pct <= 100 or _is_rate_context(line, m.start(), m.end()):
                continue
            found.setdefault(_norm(name), Segment(name, pct / 100))

    segs = list(found.values())
    total = sum(s.share for s in segs)
    return SegmentSet(segments=segs,
                      complete=len(segs) >= MIN_SEGMENTS and abs(total - 1.0) <= SUM_TOLERANCE,
                      source_line=line[:400], method="prose", corroborated=True,
                      # 서술문에는 소제목 문맥이 없다. 0 으로 두면 매출 대조에
                      # 실패한 표에도 밀린다. 회사가 문장에 직접 밝힌 % 다.
                      context_score=1)


# 같은 문맥 강도라면 뒷받침이 강한 방법이 이긴다.
# table_amount_unverified 는 매출 대조에 실패한 값이라 가장 낮다 —
# 다른 후보가 하나라도 있으면 그쪽을 쓴다.
_METHOD_RANK = {"prose": 3, "table_amount": 3, "table_pct": 2,
                "table_amount_unverified": 1}


def _rank(s: SegmentSet) -> tuple:
    """문맥 강도 → 방법 신뢰도 → 완전성 → 세그먼트 수 → 합의 1.0 근접도.

    문맥이 가장 강한 신호다. 매출 대조는 생각보다 약하다 — 회사 문서에는
    매출액과 우연히 비슷한 합을 갖는 표가 여럿 있다(실측: 넥센타이어의 깨진
    '주요 제품 등의 현황' 표가 대조를 통과해, 정확한 '매출실적' 표를 이겼다).
    어느 절 아래에 있는 표인가가 그 표가 무엇인지를 훨씬 잘 말해준다.

    서술문은 표가 아니라 문맥 점수를 매길 절이 없으므로 _parse_line 에서
    기본 1 을 준다. 그러지 않으면 회사가 직접 밝힌 % 가 대조 실패한 표에 밀린다.
    """
    total = sum(x.share for x in s.segments)
    # 단일 세그먼트는 정보량이 적다. 같은 문맥에서 여러 부문으로 쪼갠 표가 있으면
    # 그쪽을 쓴다(실측: 동아엘텍이 '검사장비 11%/OLED장비 89%' 를 두고
    # 단일 '검사장비 100%' 를 골랐다).
    return (s.context_score, len(s.segments) > 1, _METHOD_RANK.get(s.method, 0),
            s.complete, len(s.segments), -abs(total - 1.0))
