# BidCoinGenerator의 CLI 인터페이스. 이 파일을 실행하여 질문에 대한 답변을 생성할 수 있습니다.

from __future__ import annotations

from .generator import BidCoinGenerator
from .mock_data import get_mock_retrieval_result


def main() -> None:
    retrieval_result = get_mock_retrieval_result()
    generator = BidCoinGenerator()
    result = generator.generate(retrieval_result)

    print("\n===== 질문 =====")
    print(retrieval_result.question)

    print("\n===== 사용한 출처 =====")
    for source in result.used_sources:
        print(f"- {source}")

    print("\n===== 답변 =====")
    print(result.answer)


if __name__ == "__main__":
    main()
