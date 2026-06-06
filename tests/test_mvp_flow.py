from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from src import db
from src.config import AppConfig
from src.llm import BaseLLMClient, create_llm_client
from src.local_ai import (
    clean_transcript_text,
    cosine_similarity,
    is_probably_silent_wav,
    rank_embedding_matches,
    stable_text_embedding,
    wav_audio_stats,
    wav_audio_stats_from_bytes,
)
from src.models import ReflectionInput
from src.questions import DEFAULT_QUESTIONS
from src.scrapbook import (
    build_structured_memo,
    fallback_followup_question,
    generate_followup_questions,
    is_grounded_followup_question,
    should_suggest_followup,
)
from src.work_context import fetch_wikipedia_context, format_external_context_for_prompt


class LooseLLMClient(BaseLLMClient):
    def generate_questions(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> list[str]:
        return ["조금 더 적어볼까요?", "조금 더 적어볼까요?", "", "  어떤 감정이 남았나요?  "]

    def generate_memo(
        self,
        reflection_input: ReflectionInput,
        answers: dict[str, str],
    ) -> dict[str, object]:
        return {
            "title": "LLM이 바꾼 제목",
            "content_type": "LLM이 바꾼 유형",
            "one_line_summary": ["요약이", "리스트로 옴"],
            "impressive_points": "한 줄 포인트",
            "tags": ["책", "책", "", "성장"],
        }


class MvpFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db.DATA_DIR = Path(self.tmpdir.name)
        db.DB_PATH = db.DATA_DIR / "scrapbook.db"
        db.init_db()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_create_search_and_update_record(self) -> None:
        llm = create_llm_client("mock", AppConfig())
        reflection_input = ReflectionInput(
            content_type="영화",
            title="위키드",
            appreciation_date="2026-05-20",
            status="감상 완료",
            work_context="오즈의 두 인물이 갈라지는 관계를 중심으로 봤다.",
            spoiler_scope="입력 내용에서 추리한 감상 범위까지만",
            initial_note="관계가 갈라지는 장면이 오래 남았다.",
            free_text="초록색 이미지와 음악이 감정을 강하게 남겼다.",
            audio_transcript="무대가 끝난 뒤에도 마지막 화음이 남았다.",
        )
        answers = {question: "테스트 답변" for question in DEFAULT_QUESTIONS}

        followups = generate_followup_questions(llm, reflection_input, answers)
        memo = build_structured_memo(llm, reflection_input, answers)
        record_id = db.save_record(
            reflection_input=reflection_input,
            questions=[*DEFAULT_QUESTIONS, *followups],
            answers=answers,
            memo=memo,
            edited_memo=memo,
            provider="mock",
            audio_segments=[{"start": 0.0, "end": 1.5, "text": "무대가 끝난 뒤에도"}],
            external_context={
                "source": "Wikipedia EN",
                "title": "Wicked",
                "page_url": "https://en.wikipedia.org/wiki/Wicked",
                "extract": "Wicked is a musical fantasy work.",
            },
        )

        self.assertEqual(len(db.list_records("위키드")), 1)
        self.assertEqual(len(db.list_records("오즈")), 1)
        self.assertEqual(len(db.list_records("초록색")), 1)
        self.assertEqual(len(db.list_records(content_type="영화")), 1)
        self.assertEqual(len(db.list_records(content_type=["영화", "책"])), 1)
        self.assertEqual(len(db.list_records(content_type=["책"])), 0)
        self.assertEqual(len(db.list_records(status="감상 완료")), 1)
        self.assertEqual(len(db.list_records(status=["감상 완료", "감상 중"])), 1)
        self.assertEqual(len(db.list_records(status=["감상 중"])), 0)
        self.assertEqual(len(db.list_records(content_type="책")), 0)

        edited = dict(memo)
        edited["one_line_summary"] = "수정된 한 줄 요약"
        db.update_record(record_id=record_id, edited_memo=edited)

        record = db.get_record(record_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["edited_memo"]["one_line_summary"], "수정된 한 줄 요약")
        self.assertEqual(record["audio_transcript"], "무대가 끝난 뒤에도 마지막 화음이 남았다.")
        self.assertEqual(record["audio_segments"][0]["text"], "무대가 끝난 뒤에도")
        self.assertEqual(record["work_context"], "오즈의 두 인물이 갈라지는 관계를 중심으로 봤다.")
        self.assertEqual(record["external_context"]["source"], "Wikipedia EN")
        self.assertIn("Wicked", record["external_context"]["title"])
        self.assertEqual(record["spoiler_scope"], "입력 내용에서 추리한 감상 범위까지만")

    def test_silent_wav_detection(self) -> None:
        path = Path(self.tmpdir.name) / "silent.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)

        stats = wav_audio_stats(path)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.peak, 0)
        self.assertEqual(stats.duration_seconds, 1.0)
        self.assertEqual(wav_audio_stats_from_bytes(path.read_bytes()), stats)
        self.assertTrue(is_probably_silent_wav(path))

    def test_transcript_subtitle_hallucination_cleanup(self) -> None:
        self.assertEqual(
            clean_transcript_text("독자가 소설의 내용을 알고 동료들과 함께 멸망한 세계에서 살아남아 가는 길\n\n한글자막 by 한효정"),
            "독자가 소설의 내용을 알고 동료들과 함께 멸망한 세계에서 살아남아 가는 길",
        )
        self.assertEqual(clean_transcript_text("자막 제공 및 자막을 사용하였습니다."), "")

    def test_llm_outputs_are_normalized_before_storage(self) -> None:
        reflection_input = ReflectionInput(
            content_type="책",
            title="데미안",
            appreciation_date="2026-05-20",
            status="감상 중",
            initial_note="싱클레어의 혼란이 기억에 남았다.",
            free_text="아직 끝까지 읽지는 않았다.",
        )
        answers = {"질문": "불안과 성장 사이의 느낌이 있었다."}

        followups = generate_followup_questions(LooseLLMClient(), reflection_input, answers)
        memo = build_structured_memo(LooseLLMClient(), reflection_input, answers)

        self.assertEqual(followups, ["조금 더 적어볼까요?", "어떤 감정이 남았나요?"])
        self.assertEqual(memo["title"], "데미안")
        self.assertEqual(memo["content_type"], "책")
        self.assertEqual(memo["appreciation_date"], "2026-05-20")
        self.assertEqual(memo["status"], "감상 중")
        self.assertEqual(memo["one_line_summary"], "요약이\n리스트로 옴")
        self.assertEqual(memo["impressive_points"], ["한 줄 포인트"])
        self.assertEqual(memo["tags"], ["책", "성장"])
        self.assertIsInstance(memo["review_note"], str)

    def test_followup_fallback_for_short_answers(self) -> None:
        self.assertTrue(should_suggest_followup("찝찝했음"))
        self.assertFalse(should_suggest_followup(""))
        self.assertEqual(
            fallback_followup_question(1),
            "그 이유는 인물의 선택, 분위기, 내 경험 중 어디에 더 가까웠나요?",
        )
        answer = "해그리드가 해리에게 마법사라는 것을 알려주고 새로운 세계로 환대하는 장면이 설렜다."
        self.assertFalse(
            is_grounded_followup_question(
                "그 설렘은 주로 어떤 색상이나 기운으로 그려졌나요?",
                answer,
            )
        )
        self.assertTrue(
            is_grounded_followup_question(
                "그 설렘은 해방감, 환대받는 느낌, 새로운 세계에 대한 기대 중 어디에 더 가까웠나요?",
                answer,
            )
        )

    def test_embedding_and_asset_storage(self) -> None:
        llm = create_llm_client("mock", AppConfig())
        first = ReflectionInput(
            content_type="소설",
            title="프로젝트 헤일메리",
            status="감상 완료",
            initial_note="외로움보다 협력이 오래 남았다.",
        )
        second = ReflectionInput(
            content_type="영화",
            title="조용한 영화",
            status="감상 완료",
            initial_note="긴 침묵과 거리감이 남았다.",
        )
        answers = {"질문": "협력과 신뢰가 핵심이었다."}
        first_id = db.save_record(
            reflection_input=first,
            questions=["질문"],
            answers=answers,
            memo=build_structured_memo(llm, first, answers),
            edited_memo=None,
            provider="mock",
        )
        second_id = db.save_record(
            reflection_input=second,
            questions=["질문"],
            answers={"질문": "침묵이 핵심이었다."},
            memo=build_structured_memo(llm, second, {"질문": "침묵이 핵심이었다."}),
            edited_memo=None,
            provider="mock",
        )

        db.upsert_record_embedding(
            record_id=first_id,
            model="test",
            embedding=[1.0, 0.0],
            source_hash="first",
        )
        db.upsert_record_embedding(
            record_id=second_id,
            model="test",
            embedding=[0.0, 1.0],
            source_hash="second",
        )
        db.upsert_similarity_reason(
            source_record_id=first_id,
            target_record_id=second_id,
            source_hash="first",
            target_hash="second",
            provider="mock:test",
            reason="두 기록은 생존과 협력의 감정 단서가 이어져 있습니다.",
        )
        self.assertEqual(
            db.get_similarity_reason(
                source_record_id=first_id,
                target_record_id=second_id,
                source_hash="first",
                target_hash="second",
                provider="mock:test",
            ),
            "두 기록은 생존과 협력의 감정 단서가 이어져 있습니다.",
        )
        ranked = rank_embedding_matches([1.0, 0.0], db.list_record_embeddings())
        self.assertEqual(ranked[0]["record_id"], first_id)
        self.assertGreater(cosine_similarity([1.0, 0.0], [0.5, 0.0]), 0.9)

        updated_vector = stable_text_embedding("협력 신뢰 우정")
        db.upsert_record_embedding(
            record_id=first_id,
            model="test-v2",
            embedding=updated_vector,
            source_hash="updated",
        )
        stored = db.get_record_embedding(first_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["model"], "test-v2")
        self.assertEqual(stored["source_hash"], "updated")
        self.assertIsNone(
            db.get_similarity_reason(
                source_record_id=first_id,
                target_record_id=second_id,
                source_hash="first",
                target_hash="second",
                provider="mock:test",
            )
        )

        asset_id = db.add_record_asset(
            record_id=first_id,
            asset_type="tts_question",
            path="data/assets/tts/test.wav",
            metadata={"question": "q"},
        )
        self.assertGreater(asset_id, 0)
        assets = db.list_record_assets(first_id, asset_type="tts_question")
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["metadata"]["question"], "q")

    def test_work_keeps_single_record_and_updates(self) -> None:
        llm = create_llm_client("mock", AppConfig())
        first = ReflectionInput(
            content_type="book",
            title="Demian",
            status="reading",
            initial_note="early impression",
        )
        second = ReflectionInput(
            content_type="book",
            title="Demian",
            status="finished",
            initial_note="final impression",
        )
        first_id = db.save_record(
            reflection_input=first,
            questions=["q"],
            answers={"q": "a"},
            memo=build_structured_memo(llm, first, {"q": "a"}),
            edited_memo=None,
            provider="mock",
            input_mode="text",
            appreciation_stage="early",
        )
        work_id = db.get_record(first_id)["work_id"]
        self.assertEqual(db.list_record_versions(first_id), [])
        second_id = db.save_record(
            reflection_input=second,
            questions=["q"],
            answers={"q": "b"},
            memo=build_structured_memo(llm, second, {"q": "b"}),
            edited_memo=None,
            provider="mock",
            work_id=work_id,
            input_mode="voice",
            appreciation_stage="done",
        )

        works = db.list_works("Demian")
        self.assertEqual(len(works), 1)
        self.assertEqual(works[0]["record_count"], 1)
        self.assertEqual(first_id, second_id)
        records = db.get_work_records(work_id, ascending=True)
        self.assertEqual([record["id"] for record in records], [first_id])
        updated = db.get_record(first_id)
        self.assertEqual(updated["input_mode"], "voice")
        self.assertEqual(updated["appreciation_stage"], "done")
        self.assertIn("early impression", updated["initial_note"])
        self.assertIn("final impression", updated["initial_note"])
        self.assertEqual(updated["answers"]["q"], "b")
        versions = db.list_record_versions(first_id)
        self.assertEqual(len(versions), 1)
        self.assertIn("이어 기록 전", versions[0]["version_label"])
        self.assertEqual(versions[0]["input_mode"], "text")
        self.assertEqual(versions[0]["answers"]["q"], "a")
        self.assertIn("early impression", versions[0]["initial_note"])

        db.update_record(record_id=first_id, edited_memo={"one_line_summary": "manual edit"})
        self.assertEqual(len(db.list_record_versions(first_id)), 1)

        third_id = db.save_record(
            reflection_input=second,
            questions=["q"],
            answers={"q": "c"},
            memo=build_structured_memo(llm, second, {"q": "c"}),
            edited_memo=None,
            provider="mock",
            force_new_work=True,
        )
        self.assertNotEqual(db.get_record(third_id)["work_id"], work_id)
        self.assertEqual(len(db.list_works("Demian")), 2)

    def test_delete_record_removes_empty_work_and_indexes(self) -> None:
        llm = create_llm_client("mock", AppConfig())
        reflection_input = ReflectionInput(
            content_type="영화",
            title="삭제 테스트",
            status="감상 완료",
            initial_note="삭제될 기록",
        )
        record_id = db.save_record(
            reflection_input=reflection_input,
            questions=["q"],
            answers={"q": "a"},
            memo=build_structured_memo(llm, reflection_input, {"q": "a"}),
            edited_memo=None,
            provider="mock",
        )
        work_id = db.get_record(record_id)["work_id"]
        db.upsert_record_embedding(
            record_id=record_id,
            model="test",
            embedding=[1.0, 0.0],
            source_hash="delete-test",
        )
        db.add_record_asset(
            record_id=record_id,
            asset_type="tts_question",
            path="data/assets/tts/delete-test.wav",
            metadata={"question": "q"},
        )

        deleted = db.delete_record(record_id)

        self.assertIsNotNone(deleted)
        assert deleted is not None
        self.assertEqual(deleted["record_id"], record_id)
        self.assertTrue(deleted["deleted_work"])
        self.assertIsNone(db.get_record(record_id))
        self.assertIsNone(db.get_work(work_id))
        self.assertIsNone(db.get_record_embedding(record_id))
        self.assertEqual(db.list_record_assets(record_id), [])
        self.assertEqual(db.list_records("삭제 테스트"), [])

    def test_mock_llm_explains_similarity(self) -> None:
        llm = create_llm_client("mock", AppConfig())
        reason = llm.explain_similarity(
            {"title": "A", "tags": ["우주", "고립", "생존"]},
            {"title": "B", "tags": ["우주", "생존", "유머"]},
        )
        self.assertIn("우주", reason)
        self.assertIn("생존", reason)

    def test_wikipedia_context_fetch_and_prompt_format(self) -> None:
        search_response = Mock()
        search_response.raise_for_status.return_value = None
        search_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "pageid": 111,
                        "title": "Wicked",
                        "snippet": "Wicked may refer to many works",
                    },
                    {
                        "pageid": 123,
                        "title": "Wicked (2024 film)",
                        "snippet": "<span>Wicked</span> film result",
                    }
                ]
            }
        }
        extract_response = Mock()
        extract_response.raise_for_status.return_value = None
        extract_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Wicked (2024 film)",
                        "fullurl": "https://en.wikipedia.org/wiki/Wicked_(2024_film)",
                        "extract": "Wicked is a musical fantasy film.\n\nReferences\nIgnored source list",
                    }
                }
            }
        }

        config = AppConfig(
            wikipedia_store_max_chars=500,
            wikipedia_prompt_max_chars=160,
        )
        with patch("src.work_context.requests.get", side_effect=[search_response, extract_response]):
            context = fetch_wikipedia_context("Wicked", config, content_type="영화")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["title"], "Wicked (2024 film)")
        self.assertEqual(context["description"], "Wicked film result")
        self.assertIn("musical fantasy film", context["extract"])
        self.assertNotIn("Ignored source list", context["extract"])

        prompt_context = format_external_context_for_prompt(context, max_chars=160)
        self.assertIn("외부 작품 맥락", prompt_context)
        self.assertIn("사용자 감상을 대신 쓰지 말고", prompt_context)
        self.assertIn("https://en.wikipedia.org/wiki/Wicked_(2024_film)", prompt_context)

    def test_korean_wikipedia_bridge_rejects_related_person_pages(self) -> None:
        ko_search_response = Mock()
        ko_search_response.raise_for_status.return_value = None
        ko_search_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "pageid": 1,
                        "title": "J. K. 롤링",
                        "snippet": "해리 포터 소설을 쓴 작가",
                    },
                    {
                        "pageid": 2,
                        "title": "해리 포터",
                        "snippet": "조앤 롤링의 소설 시리즈",
                    },
                    {
                        "pageid": 3,
                        "title": "다니엘 래드클리프",
                        "snippet": "해리 포터 영화 시리즈의 배우",
                    },
                ]
            }
        }
        langlink_response = Mock()
        langlink_response.raise_for_status.return_value = None
        langlink_response.json.return_value = {
            "query": {
                "pages": {
                    "2": {
                        "langlinks": [{"lang": "en", "*": "Harry Potter"}],
                    }
                }
            }
        }
        extract_response = Mock()
        extract_response.raise_for_status.return_value = None
        extract_response.json.return_value = {
            "query": {
                "pages": {
                    "4": {
                        "pageid": 4,
                        "title": "Harry Potter",
                        "fullurl": "https://en.wikipedia.org/wiki/Harry_Potter",
                        "extract": "Harry Potter is a series of fantasy novels by J. K. Rowling.",
                    }
                }
            }
        }

        config = AppConfig()
        with patch(
            "src.work_context.requests.get",
            side_effect=[ko_search_response, langlink_response, extract_response],
        ):
            context = fetch_wikipedia_context(
                "해리포터",
                config,
                content_type="소설",
                creator="조앤K롤링",
            )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["title"], "Harry Potter")
        self.assertNotIn("Daniel", context["title"])

if __name__ == "__main__":
    unittest.main()
