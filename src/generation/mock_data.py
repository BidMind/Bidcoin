# 테스트하기 위한 목업데이터 생성.

from __future__ import annotations

from .schemas import ChatTurn, RetrievalResult, RetrievedContext


def get_mock_retrieval_result() -> RetrievalResult:
    contexts = [
        RetrievedContext(
            chunk_id="doc_001_chunk_01",
            text=(
                "본 사업은 이러닝 시스템 기능 고도화를 목적으로 한다. "
                "학습 콘텐츠 등록, 수정, 버전관리 기능을 제공해야 하며, "
                "관리자는 학습 이력과 진도 현황을 조회할 수 있어야 한다."
            ),
            source_file="국민연금공단_이러닝시스템.hwp",
            organization="국민연금공단",
            project_name="이러닝시스템 구축",
            summary="이러닝 콘텐츠 관리 및 학습 이력 조회 기능을 포함한 시스템 고도화 사업",
            score=0.93,
        ),
        RetrievedContext(
            chunk_id="doc_001_chunk_02",
            text=(
                "보안 요구사항으로는 사용자 권한 분리, 개인정보보호, "
                "로그 관리 및 관리자 행위 추적 기능이 요구된다."
            ),
            source_file="국민연금공단_이러닝시스템.hwp",
            organization="국민연금공단",
            project_name="이러닝시스템 구축",
            summary="보안 및 개인정보보호 요구사항 포함",
            score=0.89,
        ),
        RetrievedContext(
            chunk_id="doc_001_chunk_03",
            text=(
                "운영 측면에서는 장애 대응 체계, 백업 및 복구 방안, "
                "운영 매뉴얼 제공이 요구된다."
            ),
            source_file="국민연금공단_이러닝시스템.hwp",
            organization="국민연금공단",
            project_name="이러닝시스템 구축",
            summary="운영 및 유지보수 요구사항 포함",
            score=0.84,
        ),
    ]

    chat_history = [
        ChatTurn(role="user", content="국민연금공단 사업 문서를 찾아줘."),
        ChatTurn(role="assistant", content="국민연금공단 이러닝시스템 구축 관련 문서를 참고하겠습니다."),
    ]

    return RetrievalResult(
        question="콘텐츠 관리 요구사항과 보안 요구사항을 정리해줘.",
        contexts=contexts,
        chat_history=chat_history,
    )
