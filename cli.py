# BidCoinGenerator의 CLI 인터페이스.
# 질문을 계속 입력받아 Retrieval + Generation을 수행하고,
# quit / exit / q 입력 전까지 대화를 이어갑니다.

from __future__ import annotations

from rag_api_v2 import get_rag_context
from src.generation.generator import BidCoinGenerator
from src.generation.schemas import RetrievalResult


MAX_HISTORY_TURNS = 3   # retrieval에 넘길 최근 턴 수(질문+답변 한 쌍 = 2개 항목)


def trim_history(history: list[dict[str, str]], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
    """
    history는 [{"role": "user"|"assistant", "content": "..."}] 형태로 저장한다.
    한 턴은 user 1개 + assistant 1개이므로, 최근 max_turns 턴만 남기려면 뒤에서 max_turns * 2개를 자른다!
    """
    max_items = max_turns * 2
    return history[-max_items:]


def main() -> None:
    generator = BidCoinGenerator()
    history: list[dict[str, str]] = []

    print("BidCoin CLI")
    print("질문을 입력하세요. 종료하려면 quit / exit / q 를 입력하세요.")

    while True:
        question = input("\n질문 > ").strip()

        if not question:
            print("질문을 입력해주세요.")
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("종료합니다.")
            break

        try:
            # 최근 대화만 retrieval에 전달
            recent_history = trim_history(history)

            # Retrieval 호출
            raw_result = get_rag_context(question, recent_history)

            # Retrieval 결과를 Generation에서 기대하는 스키마로 검증
            retrieval_result = RetrievalResult.model_validate(raw_result)

            # Generation 호출
            result = generator.generate(retrieval_result)

            # 출력
            print("\n===== 질문 =====")
            print(retrieval_result.question)

            print("\n===== 사용한 출처 =====")
            if result.used_sources:
                for source in result.used_sources:
                    print(f"- {source}")
            else:
                print("- 출처 없음")

            print("\n===== 답변 =====")
            print(result.answer)

            # 현재 턴을 history에 저장
            history.append({
                "role": "user",
                "content": question,
            })
            history.append({
                "role": "assistant",
                "content": result.answer,
            })

        except Exception as e:
            print("\n[ERROR] 실행 중 문제가 발생했습니다.")
            print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()