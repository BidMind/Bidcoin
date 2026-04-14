# context만드는 과정. 검색 결과를 헤더/요약/본문으로 나눠 정리해서 모델에 전달

from __future__ import annotations

from .schemas import ChatTurn, RetrievedContext


def build_history_block(chat_history: list[ChatTurn]) -> str:
    if not chat_history:
        return "없음"

    lines: list[str] = []
    for turn in chat_history:
        prefix = "사용자" if turn.role == "user" else "어시스턴트"
        lines.append(f"- {prefix}: {turn.content}")
    return "\n".join(lines)


def _context_header(context: RetrievedContext, idx: int) -> str:
    score = f"{context.score:.4f}" if context.score is not None else "N/A"
    org = context.organization or "미상"
    project = context.project_name or "미상"
    chunk_id = context.chunk_id or f"chunk_{idx}"
    return (
        f"[문서 {idx}] chunk_id={chunk_id} | score={score} | "
        f"기관={org} | 사업명={project} | 파일명={context.source_file}"
    )


def build_context_block(
    contexts: list[RetrievedContext],
    max_contexts: int = 3,
    max_chars: int = 20000,
) -> tuple[str, list[str]]:
    if not contexts:
        return "검색된 문서가 없습니다.", []

    selected = contexts[:max_contexts]
    sections: list[str] = []
    used_sources: list[str] = []
    current_chars = 0

    for idx, context in enumerate(selected, start=1):
        header = _context_header(context, idx)
        summary = f"문서 요약: {context.summary}\n" if context.summary else ""
        body = f"{header}\n{summary}본문:\n{context.text}"
        if current_chars + len(body) > max_chars and sections:
            break
        sections.append(body)
        current_chars += len(body)
        if context.source_file not in used_sources:
            used_sources.append(context.source_file)

    return "\n\n---\n\n".join(sections), used_sources
