"""리서치 다이제스트. 데이터의 '뷰'이지 산출물이 아니다 — 원천은 parquet/json."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import pandas as pd

DART_DOC = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"


def _fired(row, check_ids: list[str]) -> list[str]:
    """이 종목에서 걸린 체크. 설정으로 켜진 것만 본다."""
    if row is None:
        return []
    return [c for c in check_ids if c in row.index and bool(row[c])]


# ── 촉매 표시 ────────────────────────────────────────────────────────
def _nm(m: pd.DataFrame, ticker: str) -> str:
    return str(m.loc[ticker, "name"]) if ticker in m.index else ""


def _mag(r) -> str:
    v = getattr(r, "magnitude", None)
    return "-" if v is None or pd.isna(v) else f"{float(v):.1%}"


# 체크 이름을 그대로 쓰면 읽는 사람이 뭘 뜻하는지 모른다.
_CHK_LABEL = {
    "chk_grounded": "근거미상", "chk_not_amended": "정정있음",
    "chk_not_expired": "만료", "chk_material": "규모미달",
    "chk_no_reversal": "반대공시", "chk_numbers_agree": "숫자불일치",
}


def _rk(report_key: str) -> str:
    """공시명 축약. '주요사항보고서(전환사채권발행결정)' → '전환사채권발행결정'.

    카탈로그의 name(지분 희석)만 쓰면 유상증자와 전환사채가 같아 보인다.
    실측: 큐로셀이 같은 날 둘 다 냈는데 표에서 중복으로 읽혔다 —
    실제로는 각 9.2% 씩 **합산 18.4%** 희석이다.
    """
    s = str(report_key or "")
    if "(" in s and s.endswith(")"):
        return s[s.index("(") + 1:-1]
    return s


def _failed(r) -> str:
    """**통과한 것이 아니라 실패한 것을 쓴다.**

    '5/6' 만 보여주면 무엇이 빠졌는지 모른다. 규모 미달인 자사주와
    정정공시가 붙은 수주는 같은 5/6 이어도 성격이 전혀 다르다.
    """
    bad = [lab for c, lab in _CHK_LABEL.items()
           if hasattr(r, c) and not bool(getattr(r, c))]
    return ", ".join(bad) or "—"


def render(as_of: date, survivors: pd.DataFrame, verdicts: pd.DataFrame,
           taxonomy: dict, funnel: dict, quality: dict,
           verify_metrics: dict, checks: dict | None = None,
           max_risk_groups: int = 2, screen_name: str = "스크린",
           catalysts: pd.DataFrame | None = None,
           min_catalyst_confidence: int = 4,
           golden: dict | None = None,
           survivorship: dict | None = None) -> str:
    names = {t["id"]: t["name_ko"] for t in taxonomy["themes"]}
    # ── 촉매 ──────────────────────────────────────────────────────
    # **만료된 것은 본문에서 빼되 없었던 일로 만들지 않는다** — 부록에 남긴다.
    # 신뢰도가 낮은 것도 마찬가지다. 조용히 버리면 '촉매가 없었다'와
    # '촉매가 약했다'가 같아진다.
    cat = catalysts if catalysts is not None else pd.DataFrame()
    cat_live = cat_stale = pd.DataFrame()
    if not cat.empty:
        fresh = cat["chk_not_expired"] & (cat["confidence"] >= min_catalyst_confidence)
        cat_live, cat_stale = cat[fresh], cat[~fresh]

    def _cat_of(ticker: str) -> pd.DataFrame:
        if cat_live.empty:
            return cat_live
        return cat_live[cat_live["ticker"] == ticker]

    def _cat_mark(ticker: str) -> str:
        """테마 표에 넣을 축약. 역촉매를 **먼저** 쓴다 — 나쁜 소식이 먼저 보여야 한다."""
        g = _cat_of(ticker)
        if g.empty:
            return "-"
        neg = [f"**{r.kind}**" for r in g[g.polarity == "negative"].itertuples()]
        pos = [r.kind for r in g[g.polarity == "positive"].itertuples()]
        return " ".join(neg + pos) or "-"
    m = survivors.set_index("ticker")
    # 어떤 체크가 켜져 있었는지는 설정에서 온다 — 다이제스트에 하드코딩하면
    # 설정으로 체크를 추가해도 리포트에 안 나온다.
    check_ids = [c for c in ((checks or {}).get("enabled") or []) if c in survivors.columns]
    # 게재 축 — 검증(status)과 별개다. 검증은 통과했지만 재무 위험이 여러
    # 범주에 걸친 종목을 본문에서 빼고 아래 절로 내린다. 판정 자체를 바꾸지
    # 않는 이유는 '테마가 맞는가'와 '싸 보이는 이유가 재무에 있는가'가
    # 서로 다른 질문이기 때문이다.
    rg = (survivors.set_index("ticker")["risk_groups"]
          if "risk_groups" in survivors.columns else None)

    def _held(ticker: str) -> bool:
        return rg is not None and ticker in rg.index and int(rg[ticker]) >= max_risk_groups

    def _share(r) -> str:
        """매출비중을 **숫자와 출처**로 보여준다. 등급으로 뭉개지 않는다.

        '미확인'은 배정이 틀렸다는 뜻이 아니라 세그먼트를 공시하지 않았다는
        뜻이다. 둘을 같은 칸에 넣으면 읽는 사람이 구분할 수 없다.
        """
        actual = getattr(r, "actual_share", None)
        if actual is not None and pd.notna(actual):
            return f"{float(actual):.0%} (실측)"
        claim = getattr(r, "claimed_share", None)
        why = {"unverified": "비율표 미검증", "unavailable": "세그먼트 미공시",
               "not_found": "세그먼트 불일치", "not_claimed": "비중 미기재"}.get(
                   getattr(r, "share_evidence", ""), "미확인")
        if claim is not None and pd.notna(claim) and float(claim) > 0:
            return f"~{float(claim):.0%} (주장, {why})"
        return why
    L: list[str] = []

    # 제목을 하드코딩하면 세 스크린의 다이제스트가 전부 "딥밸류"가 된다.
    L.append(f"# {screen_name} — {as_of} (KR)\n")
    L.append(f"유니버스 {funnel.get('universe', '?')} → 하드가드 통과 "
             f"{funnel.get('eligible', '?')} → 스크린 {funnel.get('final', '?')} "
             f"→ 검증 통과 core {verify_metrics.get('verified_core', 0)}\n")

    # ── 요약 ──
    L.append("## 요약\n")
    st = verify_metrics.get("status", {})
    L.append(f"- 테마 배정 {verify_metrics.get('assignments', 0)}건 "
             f"(입증 {st.get('verified',0)} / 폐기 {st.get('rejected',0)}), "
             f"그중 core {verify_metrics.get('verified_core', 0)}건")
    # 비중 근거의 분포를 감추지 않는다. '확인됨'이 적은 것은 분류 실패가 아니라
    # 세그먼트를 공시하지 않는 회사가 많다는 사실이다.
    if se := verify_metrics.get("share_evidence"):
        L.append("- 매출비중 근거: " +
                 ", ".join(f"{k} {v}" for k, v in sorted(se.items(), key=lambda x: -x[1])))
    hr = verify_metrics.get("hallucinated_citation_rate", 0)
    L.append(f"- 인용 검증 실패율 {hr:.1%} — 원문에 없는 근거를 댄 배정 비율")
    # **신뢰도 지표를 리포트 안에 둔다.** 별도 파일에만 있으면 다이제스트를
    # 읽는 사람은 이 문서가 얼마나 맞는지 모른 채 종목만 본다.
    if golden:
        runs = golden.get("runs") or []
        base = next((r for r in runs if r.get("reproducible")), None)
        if base:
            L.append(f"- **오분류율 {base['misclassification_rate']:.1%}** "
                     f"(골든셋 {golden.get('golden_stocks', '?')}종목, "
                     f"재현 가능 기준: {base['label'].strip()}). "
                     f"라벨은 사람 검수 전이다")
    if survivorship:
        sv = survivorship
        if sv.get("delisted_since"):
            L.append(f"- ⚠ **생존편향**: {sv['as_of']} 에 상장돼 있던 "
                     f"{sv['delisted_since']}종목이 이후 폐지돼 유니버스에 없다 "
                     f"({sv['delisted_by_outcome']}). 이 시점 결과는 성과 측정에 "
                     f"쓸 수 없다")
        else:
            L.append(f"- 생존편향 없음 — {sv['as_of']} 상장 {sv['listed_then']}종목 "
                     f"중 이후 폐지 0, 재무 커버리지 {sv['coverage']:.1%}")
    if not cat.empty:
        npos = int((cat_live["polarity"] == "positive").sum()) if not cat_live.empty else 0
        nneg = int((cat_live["polarity"] == "negative").sum()) if not cat_live.empty else 0
        L.append(f"- 촉매 {npos}건 / **역촉매 {nneg}건** "
                 f"(유효·신뢰도 {min_catalyst_confidence}+ 기준, "
                 f"{cat_live['ticker'].nunique() if not cat_live.empty else 0}종목). "
                 f"만료·저신뢰 {len(cat_stale)}건은 부록 C")
    if quality:
        soft = quality.get("soft", {})
        warn = {k: v for k, v in soft.items() if v}
        if warn:
            L.append(f"- ⚠ 데이터 경고: " +
                     ", ".join(f"{k} {v}" for k, v in sorted(warn.items(), key=lambda x: -x[1])[:5]))
    if check_ids:
        fired = {c: int(survivors[c].sum()) for c in check_ids}
        hit = {k: v for k, v in sorted(fired.items(), key=lambda x: -x[1]) if v}
        L.append(f"- 재무 체크 (스크린 {len(survivors)}종목 중 걸린 수): " +
                 (", ".join(f"{k} {v}" for k, v in hit.items()) if hit else "없음"))
        off = (checks or {}).get("disabled") or []
        if off:
            L.append(f"- 꺼진 체크: {', '.join(off)}")
    L.append("")

    # 검증 결과가 비어도 부록과 면책은 반드시 렌더링한다 —
    # 조기 반환하면 '검증을 안 돌린 문서'와 '전부 폐기된 문서'가 구별되지 않는다.
    by_theme: dict[str, list] = defaultdict(list)
    held: list = []
    if not verdicts.empty:
        core = verdicts[(verdicts.status == "verified") & (verdicts.role == "core")]
        for r in core.itertuples():
            (held if _held(r.ticker) else by_theme[r.theme_id]).append(r)
    else:
        L.append("_검증을 통과한 테마 배정이 없다._\n")

    L.append("## 테마별 (원문 입증 core)\n")
    if not by_theme:
        L.append("_입증된 core 배정 없음._\n")
    for theme_id, rows in sorted(by_theme.items(), key=lambda x: -len(x[1])):
        L.append(f"### {names.get(theme_id, theme_id)} ({len(rows)}종목)\n")
        # 매출비중을 **숫자 그대로** 싣는다. 등급으로 뭉뚱그리지 않는다 —
        # 38% 인지 88% 인지는 사람이 봐야 판단할 수 있는 정보다.
        L.append("| 종목 | 촉매 | 매출비중 | PER | PBR | FCFY | 리스크 | 논거 |")
        L.append("|---|---|---|---:|---:|---:|---|---|")
        for r in rows:
            s = m.loc[r.ticker] if r.ticker in m.index else None
            per = f"{s['per']:.1f}" if s is not None and pd.notna(s["per"]) else "-"
            pbr = f"{s['pbr']:.2f}" if s is not None and pd.notna(s["pbr"]) else "-"
            fy = f"{s['fcf_yield']:.1%}" if s is not None and pd.notna(s["fcf_yield"]) else "-"
            nm = s["name"] if s is not None else ""
            risk = ", ".join(_fired(s, check_ids)) or "-"
            L.append(f"| {r.ticker} {nm} | {_cat_mark(r.ticker)} | {_share(r)} "
                     f"| {per} | {pbr} | {fy} | {risk} | {r.rationale} |")
        L.append("")
        for r in rows[:3]:
            L.append(f"> {r.ticker} 근거: 「{r.evidence_quote[:120]}」")
        L.append("")

    if held:
        L.append(f"## 재무 경고로 본문 보류 ({len(held)}종목)\n")
        L.append(f"_테마 배정은 원문 인용으로 입증됐다. 다만 재무 위험이 "
                 f"{max_risk_groups}개 범주 이상에 걸쳐 본문에서 내렸다 — "
                 f"싸 보이는 데는 이유가 있을 수 있다._\n")
        L.append("| 종목 | 테마 | PER | PBR | FCFY | 위험 범주 | 걸린 체크 |")
        L.append("|---|---|---:|---:|---:|---:|---|")
        for r in sorted(held, key=lambda x: -int(rg[x.ticker])):
            s = m.loc[r.ticker] if r.ticker in m.index else None
            per = f"{s['per']:.1f}" if s is not None and pd.notna(s["per"]) else "-"
            pbr = f"{s['pbr']:.2f}" if s is not None and pd.notna(s["pbr"]) else "-"
            fy = f"{s['fcf_yield']:.1%}" if s is not None and pd.notna(s["fcf_yield"]) else "-"
            nm = s["name"] if s is not None else ""
            L.append(f"| {r.ticker} {nm} | {names.get(r.theme_id, r.theme_id)} | "
                     f"{per} | {pbr} | {fy} | {int(rg[r.ticker])} | "
                     f"{', '.join(_fired(s, check_ids)) or '-'} |")
        L.append("")

    # ── 촉매 ──────────────────────────────────────────────────────
    # **역촉매를 먼저 쓴다.** 촉매만 보여주는 리포트는 낙관 편향을 만든다 —
    # 실측에서 주요사항보고의 41% 가 유상증자·CB 였다. 나쁜 소식이 스크롤
    # 아래에 있으면 안 읽힌다.
    if not cat.empty:
        neg = cat_live[cat_live["polarity"] == "negative"] if not cat_live.empty \
            else cat_live
        L.append(f"## 역촉매 ({len(neg)}건)\n")
        if neg.empty:
            L.append("_없음._\n")
        else:
            L.append("_공시된 사실이다. 지분 희석은 주가와 무관하게 "
                     "기존 주주 몫을 줄인다._\n")
            L.append("| 종목 | 공시 | 크기 | 공시일 | 시효 | 신뢰도 | 미충족 |")
            L.append("|---|---|---:|---|---|---:|---|")
            for r in neg.sort_values("magnitude", ascending=False).itertuples():
                L.append(f"| {r.ticker} {_nm(m, r.ticker)} | {_rk(r.report_key)} "
                         f"| {_mag(r)} | {r.occurred_at} | {r.expires_at} "
                         f"| {r.confidence}/6 | {_failed(r)} |")
            L.append("")

        pos = cat_live[cat_live["polarity"] == "positive"] if not cat_live.empty \
            else cat_live
        L.append(f"## 촉매 ({len(pos)}건)\n")
        if pos.empty:
            L.append("_없음._\n")
        else:
            L.append(f"_신뢰도는 **결정론적 6체크의 통과 수**다 — LLM 자기신고가 "
                     f"아니라 재실행해도 같다. 미충족 항목을 함께 적는다._\n")
            for kind, g in sorted(pos.groupby("kind"),
                                  key=lambda x: -len(x[1])):
                nm0 = g["name"].iloc[0]
                L.append(f"### {kind} {nm0} ({len(g)}건)\n")
                L.append("| 종목 | 공시 | 크기 | 공시일 | 시효 | 신뢰도 | 미충족 | 원문 |")
                L.append("|---|---|---:|---|---|---:|---|---|")
                for r in g.sort_values(["confidence", "magnitude"],
                                       ascending=False).head(15).itertuples():
                    L.append(f"| {r.ticker} {_nm(m, r.ticker)} | {_rk(r.report_key)} "
                             f"| {_mag(r)} | {r.occurred_at} | {r.expires_at} "
                             f"| {r.confidence}/6 | {_failed(r)} | "
                             f"[공시]({DART_DOC.format(r.rcept_no)}) |")
                if len(g) > 15:
                    L.append(f"\n_… 외 {len(g) - 15}건_")
                L.append("")

    # ── 부록 ──
    # 폐기(원문에 없는 근거)와 강등(비중이 하한 미달)은 성격이 다르다.
    # 전자는 LLM 이 지어낸 것이고, 후자는 배정 자체는 사실이나 '주력'이 아닌 것이다.
    L.append("## 부록 A. 폐기·강등\n")
    if verdicts.empty:
        L.append("_없음._\n")
    else:
        rej = verdicts[verdicts.status == "rejected"]
        dn = verdicts[(verdicts.status == "verified")
                      & (verdicts.share_evidence == "below_floor")]
        if rej.empty and dn.empty:
            L.append("_없음._\n")
        else:
            L.append("| 종목 | 테마 | 판정 | 사유 |")
            L.append("|---|---|---|---|")
            for r in rej.head(20).itertuples():
                L.append(f"| {r.ticker} | {names.get(r.theme_id, r.theme_id)} | "
                         f"폐기 | {r.reject_reason} |")
            for r in dn.head(20).itertuples():
                fl = r.flags if isinstance(r.flags, str) else "|".join(r.flags or [])
                why = next((f for f in fl.split("|")
                            if f.startswith("role_downgraded_by_actual")), fl)
                L.append(f"| {r.ticker} | {names.get(r.theme_id, r.theme_id)} | "
                         f"core→{r.role} | {why} |")
            L.append("")

    L.append("## 부록 B. 스크린 통과 전체\n")
    cols = ["ticker", "name", "sector_code", "per", "pbr", "fcf_yield"]
    L.append(survivors.reindex(columns=cols).to_markdown(index=False,
                                                         floatfmt=".2f"))
    L.append("")
    L.append("---")
    L.append("_이 문서는 리서치 아이디어 생성물이며 매수 신호가 아니다. "
             "LLM 논거는 사업 내용 추출 결과이고, 원문 인용 검증을 통과한 것만 실린다._")
    # **수치를 하드코딩하지 않는다.** 한때 '재실행 시 100% 일치'라고 박아뒀는데,
    # 표본을 늘려 96%(46/48)로 정정한 뒤에도 이 문구만 남아 리포트가 거짓말을
    # 하고 있었다. 상수를 읽어 쓰고, 테스트가 둘의 일치를 강제한다.
    from pipeline.verify.golden import REPRODUCIBILITY as _RP
    L.append("")
    L.append(f"_※ core 배정 재실행 일치율 **{_RP['core_agreement']:.0%}**"
             f"(누적 n={_RP['n_stocks']}, 표본 {_RP['samples']}회) — "
             f"이보다 작은 차이는 잡음이다. 매출비중 강등은 LLM 이 고른 "
             f"세그먼트 이름에 기대는 부분이 남아 더 흔들린다"
             f"(판정 일치 {_RP['verdict_agreement']:.0%}). 재측정 `{_RP['tool']}`._")
    L.append("")
    L.append("_※ 매출비중 '미확인'은 배정이 틀렸다는 뜻이 아니라 회사가 "
             "세그먼트를 공시하지 않았다는 뜻이다._")
    return "\n".join(L)
