"""사업보고서 원문(document.xml) 수집 + 섹션 추출.

evidence pack 의 본문 소스다. 여기서 요약하지 않는 것이 중요하다 —
LLM 에 요약본을 주면 V1 인용 검증(원문 대조)이 불가능해진다.
길이를 줄여야 할 때는 요약이 아니라 **섹션 단위 발췌**로 자르고, 무엇을 버렸는지 남긴다.

DART document.xml 은 zip 안에 DART 전용 마크업 XML 이 들어 있다.
표준 XML 파서로는 자주 깨지므로(비정합 엔티티, 중복 속성) 태그 제거 방식으로 처리한다.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

BASE = "https://opendart.fss.or.kr/api/document.xml"

# "II. 사업의 내용" 시작 지점. 로마자/아라비아, 공백 변형을 함께 받는다.
SECTION_START = re.compile(
    r"(?:^|\n)\s*(?:II|Ⅱ|2)\s*[\.\-]\s*사\s*업\s*의\s*내\s*용", re.MULTILINE)
# 다음 섹션(III. 재무에 관한 사항)에서 끊는다.
SECTION_END = re.compile(
    r"(?:^|\n)\s*(?:III|Ⅲ|3)\s*[\.\-]\s*재\s*무\s*에?\s*관?\s*한?", re.MULTILINE)

# 따옴표 안의 '>' 를 태그 종료로 오인하면 속성 조각이 본문에 새어나온다
# (실제 사례: <TITLE ... USERMARK="F-16 B"> → 본문에 'USERMARK="F-16 B">' 잔류).
# 인용부호 구간을 통째로 소비해서 막는다.
_TAG = re.compile(r"""<(?:[^>"']|"[^"]*"|'[^']*')*>""")
_WS = re.compile(r"[ \t　]+")
# 태그만 지우면 안에 든 CSS·JS 가 본문으로 남는다. 블록째 지운다.
_STYLE_BLOCK = re.compile(r"(?is)<(style|script)\b[^>]*>.*?</\1\s*>")
_NL = re.compile(r"\n{3,}")
_ROW = "\x01"          # 표 행 구분 임시 마커 (셀 사이 개행을 흡수한 뒤 개행으로 환원)


class DocumentError(RuntimeError):
    pass


@dataclass
class DocumentClient:
    raw_root: Path
    api_key: str | None = None
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        from pipeline.ingest.dart import _dotenv_key
        import os
        self.api_key = self.api_key or os.environ.get("DART_API_KEY") or _dotenv_key()
        self.raw_root = Path(self.raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def fetch(self, rcept_no: str, refresh: bool = False) -> str:
        """원문 텍스트. raw 는 접수번호별로 불변 보관."""
        p = self.raw_root / f"{rcept_no}.txt"
        if p.exists() and not refresh:
            return p.read_text(encoding="utf-8")
        if not self.api_key:
            raise DocumentError(f"DART_API_KEY 미설정이고 캐시도 없다: {p}")

        r = self.session.get(BASE, params={"crtfc_key": self.api_key,
                                           "rcept_no": rcept_no},
                             timeout=self.timeout_s)
        r.raise_for_status()
        if r.content[:2] != b"PK":
            # 에러 시 XML 로 status 를 돌려준다
            head = r.content[:300].decode("utf-8", "ignore")
            raise DocumentError(f"{rcept_no}: zip 이 아니다 — {head}")

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            blob = z.read(_main_document(z, rcept_no))

        text = _decode(blob)
        p.write_text(text, encoding="utf-8")
        return text


def _main_document(z: zipfile.ZipFile, rcept_no: str) -> str:
    """제출 zip 에는 본문 + 첨부(감사보고서 등)가 함께 들어 있다.

    본문은 `{rcept_no}.xml`, 첨부는 `{rcept_no}_00761.xml` 형태다.
    이름순 첫 파일을 집으면 첨부(감사보고서)를 사업보고서로 착각한다 — 실제로 그랬다.
    """
    xmls = [n for n in z.namelist() if n.lower().endswith(".xml")]
    if not xmls:
        raise DocumentError(f"{rcept_no}: zip 안에 xml 이 없다")
    exact = f"{rcept_no}.xml"
    for n in xmls:
        if n.rsplit("/", 1)[-1] == exact:
            return n
    # 이름 규칙이 다르면 가장 큰 파일이 본문이다
    return max(xmls, key=lambda n: z.getinfo(n).file_size)


def _decode(blob: bytes) -> str:
    for enc in ("utf-8", "euc-kr", "cp949"):
        try:
            return blob.decode(enc)
        except UnicodeDecodeError:
            continue
    return blob.decode("utf-8", "ignore")


def _tag_sub(s: str, name: str, repl: str) -> str:
    """<name ...> / </name> 를 통째로 치환. 속성 안의 '>' 도 안전하게 처리한다."""
    pat = rf"""</?{name}(?:\s(?:[^>"']|"[^"]*"|'[^']*')*)?/?>"""
    return re.sub(pat, repl, s, flags=re.IGNORECASE)


def to_plain_text(markup: str) -> str:
    """DART 마크업 → 평문. 표는 셀 구분자를 남겨 세그먼트 수치를 읽을 수 있게 한다."""
    s = markup
    # **style/script 는 태그를 지워도 내용이 남는다.** 거래소공시(단일판매·공급계약
    # 등)는 문서 앞머리에 CSS 가 1.2KB 붙어 오는데, 그대로 두면 평문의 절반이
    # '.xforms td { padding-left:0px; …}' 이 된다(실측: 2,210자 중 1,209자).
    # 그 CSS 가 LLM 입력에 실리고 V1 인용 대조의 haystack 에도 들어간다.
    s = _STYLE_BLOCK.sub(" ", s)
    # 태그 이름만 보고 자르면 속성이 본문에 남는다(<P USERMARK="F-16 B"> → 'USERMARK=...>').
    # 반드시 여는 꺾쇠부터 닫는 꺾쇠까지 통째로 소비할 것.
    # 표 셀은 TD/TH 가 대부분이고 TE 도 쓰인다. 구분자를 넣지 않으면
    # <TD>시멘트</TD><TD>656,700</TD> 가 '시멘트656,700' 으로 붙어버려
    # 표에서 세그먼트를 읽을 수 없다(실측: 파이프 행 0/62였다).
    for cell in ("TD", "TH", "TE", "TU"):
        s = _tag_sub(s, cell, " | ")
    # 행 구분은 임시 마커로 둔다. 원본은 셀 사이에도 개행을 넣기 때문에
    # 개행을 바로 넣으면 셀 하나당 한 줄이 되어 표를 행으로 읽을 수 없다.
    s = _tag_sub(s, "TR", _ROW)
    s = _tag_sub(s, "TABLE", _ROW)
    s = _tag_sub(s, "BR", "\n")
    s = _tag_sub(s, "P", "\n")
    s = _TAG.sub("", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
           .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    # 파이프 주변의 개행을 흡수해 같은 행의 셀을 한 줄로 모은다
    s = re.sub(r"[\s　]*\|[\s　]*", " | ", s)
    s = s.replace(_ROW, "\n")
    s = _WS.sub(" ", s)
    s = _NL.sub("\n\n", s)
    return "\n".join(line.strip() for line in s.splitlines()).strip()


# 이보다 짧으면 본문이 아니라 목차·표지를 잡았을 가능성이 높다
MIN_PLAUSIBLE_SECTION = 1_500

# 예산이 이만큼도 안 남았으면 잘라 넣어봐야 문맥이 끊겨 쓸모가 없다
MIN_USEFUL_SLICE = 2_000


# "II. 사업의 내용" 안의 소제목. 예산이 모자라면 우선순위가 낮은 것부터 버린다.
# '매출 및 수주상황'을 살리는 게 중요하다 — 세그먼트 매출 비중이 거기 있고,
# 그게 없으면 V3 검증이 통째로 확인 불가가 된다.
#
# 제목 앞의 괄호 접두사를 반드시 허용할 것: 제조업과 금융업을 겸영하는 회사는
# DART 가 "1. (제조서비스업)사업의 개요" 형태로 쓴다. 첫 글자를 [가-힣A-Za-z] 로
# 못박으면 이 형태가 통째로 안 잡혀서, 해당 회사는 우선순위 0~3 소제목이 하나도
# 없는 것으로 보이고 예산이 잡동사니(자금조달 실적, 유형자산 명세)로 채워진다.
_SUBHEAD = re.compile(
    r"^[ \t]*(\d{1,2})[\.\)]\s*"
    r"((?:\([^)\n]{1,20}\)[ \t]*)?[가-힣A-Za-z][가-힣A-Za-z0-9 ·,()]{1,28})\s*$",
    re.MULTILINE)

# 상호참조 판별. 본문에는 "II. 사업의 내용" 이라는 문자열이 목차·다른 섹션의
# 참조 문구로도 등장한다("Ⅱ. 사업의 내용 참조하시기 바랍니다").
# 참조 문구를 섹션 시작으로 오인하면 뒤따르는 **엉뚱한 섹션 전체**를 사업 설명으로
# 넘기게 된다(실측: 033270 이 '외부감사에 관한 사항' 을 사업 내용으로 제출).
# '기타 참고사항' 같은 정상 소제목을 죽이지 않도록 '참고하-' 형태만 잡는다.
_XREF = re.compile(r"참조|참고하")
_XREF_WINDOW = 80
_SUB_PRIORITY = [
    (("매출", "수주"), 0),
    (("사업의 개요", "개요"), 1),
    (("주요 제품", "제품", "서비스"), 2),
    (("기타 참고",), 3),
]
_SUB_DEFAULT = 4


def _sub_priority(title: str) -> int:
    for keys, pri in _SUB_PRIORITY:
        if any(k in title for k in keys):
            return pri
    return _SUB_DEFAULT


def _subsections(body: str) -> list[tuple[int, str, str, int]]:
    """(우선순위, 제목, 본문, 원문시작위치) 목록을 우선순위 순으로. 없으면 빈 목록.

    원문 위치를 함께 돌려주는 이유: 예산 채우기는 우선순위 순으로 하되
    **출력은 원문 순서로** 되돌려야 한다. 우선순위 순으로 이어붙이면
    매출표가 사업 설명보다 앞에 오는 뒤죽박죽 문서가 LLM 에 들어간다.
    """
    heads = list(_SUBHEAD.finditer(body))
    if len(heads) < 2:
        return []
    parts = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        title = m.group(2).strip()
        parts.append((_sub_priority(title), title,
                      body[m.start():end].strip(), m.start()))
    return sorted(parts, key=lambda x: (x[0], x[3]))


@dataclass
class Excerpt:
    text: str
    found_section: bool
    truncated_chars: int
    total_chars: int
    suspiciously_short: bool = False   # 목차를 잡았을 가능성
    dropped_sections: list[str] = field(default_factory=list)
    xref_skipped: int = 0              # 상호참조로 판정해 버린 후보 수
    section_start: int = -1            # 고른 후보의 원문 위치(디버깅용)


def _is_xref(text: str, m0: re.Match) -> bool:
    """이 매칭이 섹션 시작이 아니라 다른 섹션에서의 참조 문구인가."""
    return bool(_XREF.search(text[m0.end(): m0.end() + _XREF_WINDOW]))


def _pick_section(text: str) -> tuple[str | None, int, int]:
    """(본문, 시작위치, 상호참조로 버린 수).

    후보 선정을 '가장 긴 것'으로 하면 안 된다. 목차 항목을 피하려고 넣은 규칙인데,
    상호참조 문구 뒤에 붙은 **엉뚱한 섹션**이 진짜 섹션보다 길면 그쪽이 이긴다
    (실측: 033270 은 진짜 26k 를 두고 참조문구 뒤 36k 짜리 회계감사인 섹션을 골랐다).

    상호참조를 먼저 걸러낸 뒤, 소제목이 실제로 있는 것을 우선하고
    같은 조건이면 긴 쪽을 고른다. 목차 항목에는 소제목이 없다.
    """
    best, best_key, start_at, skipped = None, None, -1, 0
    for m0 in SECTION_START.finditer(text):
        if _is_xref(text, m0):
            skipped += 1
            continue
        m1 = SECTION_END.search(text, m0.end())
        cand = text[m0.start(): m1.start() if m1 else len(text)]
        key = (len(_SUBHEAD.findall(cand)) > 1, len(cand))
        if best_key is None or key > best_key:
            best, best_key, start_at = cand, key, m0.start()
    return best, start_at, skipped


def extract_business_section(markup: str, max_chars: int = 12_000) -> Excerpt:
    """'II. 사업의 내용' 발췌.

    섹션을 못 찾으면 원문 앞부분을 쓰되 found_section=False 로 남긴다 —
    엉뚱한 부분을 사업 설명이라고 LLM 에 넘긴 사실이 드러나야 한다.
    """
    text = to_plain_text(markup)
    total = len(text)

    # "II. 사업의 내용" 은 목차에도, 다른 섹션의 참조 문구에도 나온다.
    # 어느 쪽을 집어도 found_section 은 True 라 정상처럼 보이므로 후보 선정이 중요하다.
    best, start_at, xref = _pick_section(text)
    found = best is not None
    body = best if found else text

    body = _NL.sub("\n\n", body).strip()
    short = found and len(body) < MIN_PLAUSIBLE_SECTION
    if len(body) <= max_chars:
        return Excerpt(body, found, 0, total, short,
                       xref_skipped=xref, section_start=start_at)

    # 앞에서부터 자르면 '4. 매출 및 수주상황'(세그먼트 비중이 있는 곳)이 날아간다.
    # 요약은 하지 않되, 소제목 단위로 중요한 것부터 예산을 채운다.
    #
    # 안 들어가면 통째로 버리는 방식은 쓰지 않는다. '사업의 개요' 하나가 예산보다
    # 크다는 이유로 탈락하고 그 자리를 '원재료 및 생산설비'가 채우는 일이 실제로
    # 있었다(코메론·제이에스코퍼레이션: 개요 13.7k > 예산 12k).
    # 잘린 개요가 온전한 원재료 명세보다 낫다.
    kept, dropped, used = [], [], 0
    for pri, title, chunk, pos in _subsections(body):
        room = max_chars - used
        if len(chunk) <= room:
            kept.append((pos, chunk))
            used += len(chunk)
        elif room >= MIN_USEFUL_SLICE:
            kept.append((pos, chunk[:room]))
            used += room
            dropped.append(f"{title}(일부만)")
        else:
            dropped.append(title)
    if not kept:                                   # 소제목이 없으면 종전대로 앞부분
        return Excerpt(body[:max_chars], found, len(body) - max_chars, total, short,
                       xref_skipped=xref, section_start=start_at)

    # 예산은 우선순위로 채우되 **출력은 원문 순서로** 되돌린다.
    kept.sort(key=lambda x: x[0])
    text_out = "\n\n".join(c for _, c in kept)
    return Excerpt(text_out, found, len(body) - len(text_out), total, short,
                   dropped_sections=dropped, xref_skipped=xref,
                   section_start=start_at)
