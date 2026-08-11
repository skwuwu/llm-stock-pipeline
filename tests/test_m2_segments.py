"""세그먼트 표 파싱. 전부 실제 사업보고서에서 관측한 형태로 고정한다.

틀린 세그먼트는 없는 것보다 나쁘다 — V3 가 엉뚱한 수치로 core 를 확정하거나
정상 배정을 강등시킨다. 그래서 이 테스트는 재현율보다 정밀도를 지킨다.
"""

from __future__ import annotations

import pytest

from pipeline.enrich.segments import parse_segments
from pipeline.ingest.dart_document import to_plain_text

# 성신양회 FY2025 실제 형태: 금액만 있고 비율 컬럼이 없다.
# 합 1,216,265백만원 = 1조 2,163억원 (회사가 서술문에 쓴 값과 일치)
TABLE_AMOUNT = """가. 매출실적

| 사업부문 | | 제품 | | 구체적용도 | | 제60기 | | 제59기 | | 제58기 |

| 시 멘 트 | | 시멘트 | | 내수 | | 656,676 | | 739,475 | | 783,156 |

| 레 미 콘 | | 레미콘 | | 내수 | | 140,694 | | 175,652 | | 185,361 |

| 무 역 | | 에너지,금속 | | 수출 | | 398,569 | | 225,567 | | 132,934 |

| 기 타 | | 기타 | | 내수 | | 20,326 | | 21,939 | | 11,831 |
"""
REVENUE = 1_216_265_000_000.0        # 백만원 표 × 1e6


def test_amount_table_normalized_to_shares():
    """비율 컬럼이 없으면 금액을 정규화한다. 회사가 서술문에 밝힌 값과 일치해야 한다."""
    ss = parse_segments(TABLE_AMOUNT, revenue=REVENUE)
    assert ss.method == "table_amount"
    assert ss.share_of("시멘트") == pytest.approx(0.540, abs=0.005)
    assert ss.share_of("레미콘") == pytest.approx(0.116, abs=0.005)
    assert ss.share_of("무역") == pytest.approx(0.328, abs=0.005)


def test_amount_table_uses_current_period_column():
    """제60기(당기)를 써야 한다. 제59기를 쓰면 비중이 달라진다."""
    ss = parse_segments(TABLE_AMOUNT, revenue=REVENUE)
    # 제59기 기준이면 시멘트가 62.9% 가 된다
    assert ss.share_of("시멘트") < 0.60


# 매출실적 절 밖에 있는, 매출표처럼 생긴 표(단가표·거래처표를 모사).
# 강한 문맥이 없으므로 매출 대조 실패는 곧 폐기여야 한다.
TABLE_NO_STRONG_CONTEXT = TABLE_AMOUNT.replace("가. 매출실적", "가. 주요 제품 가격변동")


def test_amount_table_dropped_when_revenue_mismatches_without_strong_context():
    """단가표·거래처표를 매출 구성으로 읽지 않기 위한 자기검증.
    어휘 목록이 아니라 이미 신뢰하는 매출액으로 거른다."""
    ss = parse_segments(TABLE_NO_STRONG_CONTEXT, revenue=99_000_000_000_000.0)
    assert not ss.segments


def test_revenue_mismatch_under_strong_context_is_kept_but_uncorroborated():
    """'가. 매출실적' 아래 표는 대조에 실패해도 버리지 않는다.

    매출실적표에는 '(내부거래포함)' 처럼 연결매출과 정의가 다른 것이 흔하다.
    다만 뒷받침이 없으므로 V3 가 이 수치로 강등해서는 안 된다 —
    그 보장은 corroborated=False 로 표현된다.
    """
    ss = parse_segments(TABLE_AMOUNT, revenue=99_000_000_000_000.0)   # 100배 차이
    assert ss.method == "table_amount_unverified"
    assert ss.corroborated is False

    from pipeline.verify.layers import _has_segments, _has_uncorroborated_segments
    pack = {"segments": ss.as_dict()}
    assert not _has_segments(pack)              # V3 강등 근거로 쓰이지 않는다
    assert _has_uncorroborated_segments(pack)   # '없음'과는 구분된다


def test_corroborated_table_beats_unverified_one():
    """대조에 성공한 표가 있으면 그쪽을 쓴다."""
    both = TABLE_NO_STRONG_CONTEXT + "\n\n" + TABLE_AMOUNT
    ss = parse_segments(both, revenue=REVENUE)
    assert ss.method == "table_amount"
    assert ss.corroborated is True


def test_segment_names_whitespace_normalized():
    """원문 '시 멘 트' 를 '시멘트' 로 정규화해야 LLM 이 쓴 이름과 매칭된다."""
    ss = parse_segments(TABLE_AMOUNT, revenue=REVENUE)
    assert "시멘트" in {s.name for s in ss.segments}


# 티에이치엔 실제 형태: 경쟁사 점유율. 합이 정확히 100이라 비율 검증만으로는 안 걸린다.
TABLE_MARKET_SHARE = """나. 시장점유율

| 구분 | | 당사 | | 경쟁사A | | 경쟁사B |

| 티에이치엔 | | 13.0 | | 0 | | 0 |

| 경신공업 | | 46.0 | | 0 | | 0 |

| 유라코퍼레이션 | | 41.0 | | 0 | | 0 |
"""


def test_market_share_table_rejected_by_context():
    """합이 정확히 100이어도 '점유율' 문맥이면 매출 구성이 아니다."""
    assert not parse_segments(TABLE_MARKET_SHARE, revenue=REVENUE).segments


# 샘표식품 실제 오탐: 유형자산 명세를 매출 구성으로 읽었다.
TABLE_FIXED_ASSETS = """다. 매출 관련 설비 현황

| 구분 | | 소재지 | | 기초 | | 기말 |

| 건물 | | 이천 | | 1,000 | | 1,100 |

| 구축물 | | 이천 | | 200 | | 210 |

| 기계장치 | | 이천 | | 3,000 | | 3,200 |

| 차량운반구 | | 이천 | | 50 | | 55 |
"""


def test_fixed_asset_table_rejected_by_row_names():
    """문맥에 '매출'이 있어도 행 이름이 자산 계정이면 매출 구성이 아니다."""
    assert not parse_segments(TABLE_FIXED_ASSETS, revenue=4_565.0).segments


# 삼성전자 실제 형태: 서술문
PROSE = ("2024년 매출은 DX 부문이 174조 8,877억원(58.1%), DS 부문이 111조 660억원(36.9%)"
         "이며, SDC가 29조 1,578억원(9.7%), Harman은 14조 2,749억원(4.7%)입니다.")


def test_prose_sentence_parsed():
    ss = parse_segments(PROSE)
    assert ss.method == "prose"
    assert ss.share_of("DX") == pytest.approx(0.581)
    assert ss.share_of("Harman") == pytest.approx(0.047)


def test_growth_rate_not_read_as_share():
    """'전년 대비 4.6% 증가' 를 세그먼트 4.6% 로 읽으면 안 된다."""
    line = "당사의 매출액은 1조 2,163억원으로 전년 동기 대비 537억원, 4.6% 증가하였습니다."
    assert not parse_segments(line).segments


def test_growth_rate_filter_does_not_kill_the_following_sentence():
    """증감률 문장 **다음** 문장의 정상 구성비는 살려야 한다.
    실측: 줄 단위로 걸렀더니 성신양회의 시멘트 54% 가 통째로 사라졌다."""
    line = ("당사의 매출액은 1조 2,163억원으로 전년 동기 대비 4.6% 증가하였습니다. "
            "매출액은 시멘트 6,567억원(54.0%), 레미콘 1,407억원(11.6%), "
            "무역 3,986억원(32.8%), 기타 203억원(1.6%)으로 구분됩니다.")
    ss = parse_segments(line)
    assert ss.share_of("시멘트") == pytest.approx(0.540)
    assert len(ss.segments) == 4


# 표 셀 구분자 — 이게 없으면 표를 행으로 읽을 수 없다
MARKUP_TABLE = """<TABLE><TBODY>
<TR><TH>사업부문</TH>
<TH>매출액</TH></TR>
<TR><TD>시멘트</TD>
<TD>656,676</TD></TR>
</TBODY></TABLE>"""


def test_td_th_cells_become_separated_row():
    """DART 표 셀은 TD/TH 이고 셀 사이에 개행이 들어간다.
    구분자 없이 태그만 지우면 '시멘트656,676' 으로 붙고, 개행을 그대로 두면
    셀 하나당 한 줄이 되어 어느 쪽이든 표를 읽을 수 없다."""
    t = to_plain_text(MARKUP_TABLE)
    rows = [l for l in t.splitlines() if l.count("|") >= 2]
    assert rows, "표 행이 하나도 안 만들어졌다"
    assert any("시멘트" in r and "656,676" in r for r in rows)
    assert "시멘트656,676" not in t


def test_empty_input_returns_empty_set():
    ss = parse_segments("")
    assert not ss.segments and not ss.complete and ss.method == ""



# ── 입증 등급 ───────────────────────────────────────────────────────
def test_amount_table_corroborated_by_revenue():
    """금액 합이 매출액과 맞으면 독립적으로 뒷받침된 수치다."""
    ss = parse_segments(TABLE_AMOUNT, revenue=REVENUE)
    assert ss.corroborated


def test_prose_is_corroborated():
    """회사가 문장에 직접 % 를 밝힌 값."""
    assert parse_segments(PROSE).corroborated


def test_percent_table_is_not_corroborated():
    """비율 컬럼은 대조할 수단이 없다. 쓰되 '입증됨'으로 부르지 않는다."""
    tbl = """가. 매출실적

| 사업부문 | | 비율 |

| 가사업 | | 60.0 |

| 나사업 | | 40.0 |
"""
    ss = parse_segments(tbl, revenue=1_000.0)
    assert ss.method == "table_pct"
    assert ss.segments and not ss.corroborated


def test_connector_phrase_between_segment_and_percent():
    """'기계사업부문은 연결매출의 83.6%' 처럼 부문과 숫자 사이에 연결어구가 온다.
    실측: 화천기공 — LLM 은 이 문장을 인용했는데 파서는 못 읽고 있었다."""
    line = ("2025년 기준, 기계사업부문은 연결매출의 83.6%에 해당하며, "
            "소재사업부문은 16.4%에 해당합니다.")
    ss = parse_segments(line)
    assert ss.share_of("기계") == pytest.approx(0.836)
    assert ss.share_of("소재") == pytest.approx(0.164)


# ── rowspan 매출실적표 ───────────────────────────────────────────────
# 넥센타이어·KG스틸·팜스토리 실제 형태. 사업부문 셀이 rowspan 으로 묶여 있어
# 이어지는 행에는 그 셀이 **아예 없다**(빈 셀이 아니라 없다).
# 왼쪽 기준 열 정렬이 성립하지 않으므로 숫자를 오른쪽에서부터 세야 한다.
TABLE_ROWSPAN = """가. 매출실적

| (단위 : 백만원) |

| 사업부문 | | 매출유형 | | 품목 | | 2025년 | | 2024년 |

| 사료사업부 | | 제품매출 | | 배합사료 | | 822,643 | | 857,951 |

| 상품매출 | | 배합사료 | | 1,544 | | 1,266 |

| 소 계 | | 824,187 | | 859,217 |

| 육가공사업부 | | 제품매출 | | 지육 | | 595,615 | | 562,860 |

| 상품매출 | | 정육 | | 94,350 | | 82,554 |

| 소 계 | | 689,965 | | 645,414 |
"""
ROWSPAN_REVENUE = 1_514_152_000_000.0     # 824,187 + 689,965 백만원


def test_rowspan_table_folds_into_business_segments():
    ss = parse_segments(TABLE_ROWSPAN, revenue=ROWSPAN_REVENUE)
    assert ss.method == "table_amount"
    assert ss.corroborated is True
    assert {s.name for s in ss.segments} == {"사료사업부", "육가공사업부"}
    assert ss.share_of("사료사업부") == pytest.approx(0.544, abs=0.005)


def test_rowspan_continuation_row_is_not_a_segment():
    """'상품매출' 은 매출유형이지 사업부문이 아니다. 열 정렬을 가정하면 이게 잡힌다."""
    ss = parse_segments(TABLE_ROWSPAN, revenue=ROWSPAN_REVENUE)
    assert "상품매출" not in {s.name for s in ss.segments}


def test_header_row_is_found_when_unit_line_comes_first():
    """DART 는 '(단위 : 백만원)' 을 별도 행으로 먼저 넣는다.
    첫 행만 헤더로 보면 정상 매출실적표가 통째로 버려진다."""
    from pipeline.enrich.segments import _header_text
    block = [["(단위 : 백만원)"], ["사업부문", "매출유형", "2025년"]]
    assert "사업부문" in _header_text(block)


def test_grand_total_row_does_not_become_a_segment():
    """마지막 그룹에 회사 전체 매출이 들어가면 비중이 통째로 망가진다."""
    body = TABLE_ROWSPAN + "\n| 합 계 | | 1,514,152 | | 1,504,631 |\n"
    ss = parse_segments(body, revenue=ROWSPAN_REVENUE)
    assert ss.share_of("육가공사업부") == pytest.approx(0.456, abs=0.005)


def test_first_subtotal_wins_when_a_group_has_several():
    """수출 소계·내수 소계가 잇따르면 마지막 작은 값이 사업부문 매출로 남는다."""
    from pipeline.enrich.segments import _row_groups
    block = [["사업부문", "품목", "2025년"],
             ["줄자사업", "줄자", "수출", "55,199"],
             ["내수", "11,218"],
             ["합 계", "66,417"],
             ["압연사업", "압연", "수출", "1,434"],
             ["합 계", "1,434"]]
    vals = {n: v for n, _p, v, _e in _row_groups(block)}
    assert vals["줄자사업"] == pytest.approx(66_417)


# ── 퇴화 파싱 거부 ───────────────────────────────────────────────────
def test_repeated_group_names_are_rejected():
    """같은 이름이 반복되면 행 묶기가 실패한 것이다 (실측: 한전 수주현황표)."""
    from pipeline.enrich.segments import Segment, SegmentSet, _is_degenerate
    ss = SegmentSet(segments=[Segment("ICT설비유지보수등", 0.5),
                              Segment("ICT설비유지보수등", 0.5)])
    assert _is_degenerate(ss)


def test_region_split_is_not_a_segment():
    """내수/수출은 지역 구분이라 테마 검증에 쓸 수 없다."""
    from pipeline.enrich.segments import Segment, SegmentSet, _is_degenerate
    assert _is_degenerate(SegmentSet(segments=[Segment("내수", 0.9),
                                               Segment("수출", 0.1)]))


def test_all_but_one_segment_at_zero_is_rejected():
    """하나가 사실상 100%, 나머지가 0% 면 집계가 실패한 것이다."""
    from pipeline.enrich.segments import Segment, SegmentSet, _is_degenerate
    assert _is_degenerate(SegmentSet(segments=[Segment("A", 0.999),
                                               Segment("B", 0.001)]))


def test_pnl_line_items_are_not_segments():
    """'매출액 67% / 매출총이익 33%' 는 사업부문이 아니다 (실측: HS애드)."""
    from pipeline.enrich.segments import Segment, SegmentSet, _looks_non_revenue
    assert _looks_non_revenue(SegmentSet(segments=[Segment("매출액", 0.67),
                                                   Segment("매출총이익", 0.33)]))


def test_ambiguous_name_lookup_returns_none():
    """여러 세그먼트에 걸리면 하나를 골라 돌려주면 안 된다 —
    V3 가 엉뚱한 비중으로 강등한다."""
    from pipeline.enrich.segments import Segment, SegmentSet
    ss = SegmentSet(segments=[Segment("반도체장비", 0.6), Segment("반도체소재", 0.4)])
    assert ss.share_of("반도체") is None
    assert ss.share_of("반도체장비") == pytest.approx(0.6)


def test_prose_outranks_uncorroborated_table():
    """회사가 문장에 직접 밝힌 % 가 매출 대조에 실패한 표에 져서는 안 된다."""
    prose = "당사의 매출은 기계사업부문 84%, 소재사업부문 16%로 구성됩니다."
    ss = parse_segments(prose + "\n\n" + TABLE_NO_STRONG_CONTEXT, revenue=REVENUE)
    assert ss.method == "prose"


# ── 단일 사업부문 회사 ───────────────────────────────────────────────
TABLE_SINGLE = """가. 매출실적

| (단위 : 백만원) |

| 사업부문 | | 품목 | | 2025년 | | 2024년 |

| 유압기기 | | 유압실린더 | | 내수 | | 300,000 | | 280,000 |

| 수출 | | 200,000 | | 190,000 |

| 합 계 | | 500,000 | | 470,000 |
"""


def test_single_segment_kept_when_revenue_corroborates():
    """사업부문이 하나뿐인 회사는 실재하고, V3 에게는 가장 쓸모 있는 정보다
    ('사실상 100% 이 사업' 이면 core 배정을 그대로 확인해준다)."""
    ss = parse_segments(TABLE_SINGLE, revenue=500_000_000_000.0)
    assert [s.name for s in ss.segments] == ["유압기기"]
    assert ss.corroborated is True
    assert ss.share_of("유압기기") == pytest.approx(1.0)


def test_single_segment_dropped_without_revenue_corroboration():
    """그룹이 하나면 '접기가 실패해 뭉친 것'과 구별할 수단이 매출 대조뿐이다."""
    assert not parse_segments(TABLE_SINGLE, revenue=9_000_000_000_000.0).segments


def test_single_prose_percent_is_not_a_segment_set():
    """문장 하나에서 % 하나만 잡힌 것은 대개 오독이다
    (실측: 한전 '기타' 100%, 폴라리스AI파마 '내수와 수출' 100%)."""
    assert not parse_segments("도시가스 부문이 97% 이상을 차지하고 있습니다.").segments


def test_multi_segment_candidate_beats_single_one():
    """정보량이 적은 쪽이 이기면 안 된다
    (실측: 동아엘텍이 '검사장비 11%/OLED장비 89%' 를 두고 단일 100% 를 골랐다)."""
    multi = """가. 매출실적

| 사업부문 | | 비율 |

| 검사장비부문 | | 11.0 |

| OLED제조장비부문 | | 89.0 |
"""
    ss = parse_segments(TABLE_SINGLE + "\n\n" + multi, revenue=500_000_000_000.0)
    assert len(ss.segments) == 2


# ── 문맥 배제 ────────────────────────────────────────────────────────
def test_production_volume_table_rejected_by_context():
    """생산실적표는 단위가 톤·대라 금액이 아니다. 그런데 배율 후보(1e3/1e6/1e8)
    중 하나가 매출과 우연히 맞을 수 있어 매출 대조만으로는 안 걸린다
    (실측: 팜스토리 사료 1,598,668톤 × 1e6 이 매출 1.45조와 10% 차이로 통과)."""
    tbl = """(2) 생산실적 및 가동률 ① 생산실적

| 사업부문 | | 품목 | | 2025년 |

| 사료사업부 | | 배합사료 | | 1,598,668 |

| 육가공사업부 | | 지육 | | 687 |

| 가금사업부 | | 닭고기 | | 13,175 |
"""
    assert not parse_segments(tbl, revenue=1_446_577_944_574.0).segments


def test_minor_residual_named_상품_does_not_kill_the_table():
    """'전력기기 65% / 상품 11% / 신재생 24%' 에서 '상품' 을 이유로 표 전체를
    버리면 정상 파싱을 잃는다(실측: 피에스텍). 지역·유형 이름이 세그먼테이션
    **축**일 때만 거부한다."""
    from pipeline.enrich.segments import Segment, SegmentSet, _is_degenerate
    ok = SegmentSet(segments=[Segment("전력기기", 0.65), Segment("상품", 0.11),
                              Segment("신재생", 0.24)])
    assert not _is_degenerate(ok)
    axis = SegmentSet(segments=[Segment("내수", 0.915), Segment("수출", 0.085)])
    assert _is_degenerate(axis)
