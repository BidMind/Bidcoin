"""
tables.py — eval.py 가 생성한 JSON 결과를 pandas 테이블로 시각화

사용법:
    from src.evaluation.tables import show
    show("eval_results_20250418_120000.json")

    # 개별 테이블만 필요할 때
    from src.evaluation.tables import load_json, judge_table, basic_table
    data = load_json("eval_results_...json")
    judge_table(data["judge"])   # LLM-as-Judge 지표 테이블
    basic_table(data["basic"])   # 검색/생성 기본 지표 테이블
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd


# ── 컬럼 표시명 매핑 ─────────────────────────────────────────────────────────

_JUDGE_COLS = {
    "relevance":         "Relevance",
    "faithfulness":      "Faithfulness",
    "correctness":       "Correctness",
    "completeness":      "Completeness",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "judge_score":       "Judge Score",
}

_BASIC_COLS = {
    "hit_rate":    "Hit Rate",
    "precision":   "Precision",
    "recall":      "Recall",
    "mrr":         "MRR",
    "ndcg":        "NDCG",
    "f1":          "F1",
    "faithfulness": "Faithfulness",
}

_CATEGORY_ORDER = [
   "의미기반", "키워드기반", "복합모호",
]


# ── JSON 로드 ────────────────────────────────────────────────────────────────

def load_json(path: Union[str, Path]) -> dict:
    """
    eval.py 가 생성한 JSON 파일을 로드하고 {"basic": [...], "judge": [...]} 형태로 반환.

    - --basic 만 실행했으면 {"basic": records, "judge": []}
    - --judge 만 실행했으면 {"basic": [], "judge": records}
    - 둘 다 실행했으면 {"basic": [...], "judge": [...]}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        # 단일 모드 — 키로 판별
        sample_eval = raw[0].get("evaluation", {}) if raw else {}
        if "relevance" in sample_eval or "correctness" in sample_eval:
            return {"basic": [], "judge": raw}
        else:
            return {"basic": raw, "judge": []}

    # 두 모드 모두 포함된 경우
    return {
        "basic": raw.get("basic", []),
        "judge": raw.get("judge", []),
    }


# ── 카테고리별 집계 헬퍼 ─────────────────────────────────────────────────────

def _aggregate_by_category(records: list[dict], metric_keys: list[str]) -> pd.DataFrame:
    rows = []
    for r in records:
        ev = r.get("evaluation", {})
        row = {"category": r.get("category", "unknown")}
        for k in metric_keys:
            row[k] = ev.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["category"])

    agg = df.groupby("category")[metric_keys].mean()

    # 정해진 카테고리 순서 적용 (없는 카테고리는 뒤에)
    ordered = [c for c in _CATEGORY_ORDER if c in agg.index]
    extra   = [c for c in agg.index if c not in _CATEGORY_ORDER]
    return agg.loc[ordered + extra]


# ── judge 테이블 ─────────────────────────────────────────────────────────────

def judge_table(records: list[dict], show_judge_score: bool = True) -> pd.DataFrame:
    """
    LLM-as-Judge 지표를 카테고리별로 집계한 테이블 반환.

    Rows    : 카테고리 (fact / condition / …)
    Columns : Relevance / Faithfulness / Correctness / Completeness /
              Context Precision / Context Recall [/ Judge Score]
    """
    metric_keys = list(_JUDGE_COLS.keys())
    if not show_judge_score:
        metric_keys = [k for k in metric_keys if k != "judge_score"]

    # judge_score 는 evaluation 안에 없으므로 별도 계산
    base_keys = [k for k in metric_keys if k != "judge_score"]
    df = _aggregate_by_category(records, base_keys)

    if show_judge_score and "judge_score" in metric_keys:
        df["judge_score"] = df[base_keys].mean(axis=1)

    df = df.rename(columns={k: v for k, v in _JUDGE_COLS.items() if k in df.columns})
    df.index.name = "Category"
    return df


# ── basic 테이블 ─────────────────────────────────────────────────────────────

def basic_table(records: list[dict], transpose: bool = False) -> pd.DataFrame:
    """
    기본 검색/생성 지표를 카테고리별로 집계한 테이블 반환.

    기본(transpose=False):
        Rows    : 카테고리
        Columns : Hit Rate / Precision / Recall / MRR / NDCG / F1 / Faithfulness

    transpose=True (스크린샷 하단 형식):
        Rows    : 지표명
        Columns : 카테고리
    """
    metric_keys = list(_BASIC_COLS.keys())
    df = _aggregate_by_category(records, metric_keys)
    df = df.rename(columns={k: v for k, v in _BASIC_COLS.items() if k in df.columns})
    df.index.name = "Category"

    if transpose:
        df = df.T
        df.index.name = "Metric"
    return df


# ── 전체 요약 테이블 ─────────────────────────────────────────────────────────

def summary_table(data: dict) -> pd.DataFrame:
    """
    basic + judge 결과를 하나의 요약 테이블로 합산.
    카테고리별 전체 지표를 한눈에 확인.
    """
    parts = []

    if data.get("basic"):
        b = basic_table(data["basic"])
        parts.append(b)

    if data.get("judge"):
        j = judge_table(data["judge"])
        parts.append(j)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, axis=1)


# ── 전체 평균 행 추가 헬퍼 ──────────────────────────────────────────────────

def with_mean_row(df: pd.DataFrame, label: str = "Overall") -> pd.DataFrame:
    """테이블 맨 아래에 전체 평균 행을 추가해 반환."""
    mean_row = df.mean(numeric_only=True).to_frame(label).T
    mean_row.index.name = df.index.name
    return pd.concat([df, mean_row])


# ── 메인 진입점 ──────────────────────────────────────────────────────────────

def show(
    path: Union[str, Path],
    *,
    transpose_basic: bool = False,
    show_mean: bool = True,
) -> None:
    """
    JSON 파일을 읽어 평가 테이블을 출력한다.

    Args:
        path            : eval.py 가 저장한 JSON 파일 경로
        transpose_basic : True면 basic 테이블을 행=지표, 열=카테고리로 전치
        show_mean       : True면 Overall 평균 행 추가
    """
    data = load_json(path)

    pd.set_option("display.float_format", "{:.6f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    if data["judge"]:
        print("\n" + "=" * 70)
        print("  LLM-as-Judge 평가  (카테고리별 평균, 0.0 ~ 1.0)")
        print("=" * 70)
        jt = judge_table(data["judge"])
        print(with_mean_row(jt) if show_mean else jt)

    if data["basic"]:
        print("\n" + "=" * 70)
        print("  기본 평가  (카테고리별 평균, 0.0 ~ 1.0)")
        print("=" * 70)
        bt = basic_table(data["basic"], transpose=transpose_basic)
        print(with_mean_row(bt) if show_mean else bt)

    if not data["judge"] and not data["basic"]:
        print("[tables.py] JSON에 평가 결과가 없습니다.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="eval.py 결과 JSON을 테이블로 출력")
    parser.add_argument("path", type=str, help="eval_results_20260418_090216.json 경로")
    parser.add_argument("--transpose", action="store_true", help="basic 테이블 행/열 전치")
    parser.add_argument("--no-mean", action="store_true", help="Overall 평균 행 제거")
    args = parser.parse_args()

    show(args.path, transpose_basic=args.transpose, show_mean=not args.no_mean)