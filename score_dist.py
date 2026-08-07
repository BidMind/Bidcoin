# 질문셋에 대해 리랭커 점수 분포를 한번에 보는 스크립트
# SCORE_THRESHOLD 결정용

import os
import sys
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
from config import OUTPUT_DIR
import argparse

from rag_api_v4 import get_rag_context
from src.evaluation.question_sets2 import get_all

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

alpha = args.alpha

print(f"[실험 설정] alpha = {alpha}")
rows = []

for item in get_all():
    question = item["question"]
    category = item.get("category", "unknown")

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            raw = get_rag_context(question, [], alpha=alpha)

    contexts = raw.get("contexts", [])

    for rank, ctx in enumerate(contexts, start=1):
        score = ctx.get("raw_score", ctx.get("score"))

        rows.append({
            "question": question,
            "category": category,
            "rank": rank,
            "score": score,                     # 분석용 점수
            "display_score": ctx.get("score"),  # 표시용 점수
            "preview": str(ctx.get("text", ctx.get("content", "")))[:300]
        })


df_scores = pd.DataFrame(rows)

if df_scores.empty:
    print("수집된 점수가 없습니다.")
    sys.exit()


print(f"\n총 수집 점수: {len(df_scores)}개")
print(f"최솟값: {df_scores['score'].min():.4f}")
print(f"최댓값: {df_scores['score'].max():.4f}")
print(f"평균:   {df_scores['score'].mean():.4f}")
print(f"중앙값: {df_scores['score'].median():.4f}")

print("\n[전체 점수 요약]")
print(df_scores["score"].describe())

print("\n[구간별 분포]")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
dist = pd.cut(
    df_scores["score"],
    bins=bins,
    include_lowest=True,
    right=False
).value_counts().sort_index()

print(dist)

print("\n[카테고리별 점수 요약]")
print(df_scores.groupby("category")["score"].describe())

print("\n[rank별 점수 요약]")
print(df_scores.groupby("rank")["score"].describe())

print("\n[질문별 top1 점수 요약]")
top1 = (
    df_scores.sort_values(["question", "score"], ascending=[True, False])
    .groupby("question")
    .head(1)
)

print(top1["score"].describe())

print("\n[threshold별 살아남는 문서 수]")
for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
    selected = (df_scores["score"] >= th).sum()
    print(f"threshold >= {th:.2f}: {selected}개 / {len(df_scores)}개")

print("\n[threshold별 context 0개가 되는 질문 수]")
total_questions = df_scores["question"].nunique()

for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
    alive_by_question = df_scores[df_scores["score"] >= th].groupby("question").size()
    zero_questions = total_questions - alive_by_question.size

    print(f"threshold >= {th:.2f}: context 0개 질문 수 = {zero_questions}개 / {total_questions}개")

output_path = OUTPUT_DIR / f"reranker_score_dist_alpha_{alpha}.csv"
df_scores.to_csv(output_path, index=False, encoding="utf-8")

print("\n저장 완료:", output_path)