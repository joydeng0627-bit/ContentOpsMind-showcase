"""ContentOpsMind evaluation framework."""

import json
import logging
import pathlib
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
from core.llm_gateway import shared_client, gateway_call, llm_task

from core.intent_recognizer import IntentRecognizer
from core.llm_response import extract_model_text

logger = logging.getLogger(__name__)


@dataclass
class IntentTestCase:
    message: str
    expected_intent: Union[str, List[str]]
    context: Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    relevance: float
    accuracy: float
    completeness: float
    helpfulness: float
    brand_consistency: float = 0.8
    compliance: float = 0.8
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean(
            [
                self.relevance,
                self.accuracy,
                self.completeness,
                self.helpfulness,
                self.brand_consistency,
                self.compliance,
            ]
        )


@dataclass
class EvalResult:
    test_id: str
    passed: bool
    scores: Dict[str, float]
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    timestamp: str
    total: int
    passed: int
    pass_rate: float
    avg_scores: Dict[str, float]
    regressions: List[str]
    recommendations: List[str]
    results: List[EvalResult]


class LLMJudge:
    JUDGE_PROMPT = """你是内容运营质量评测专家。请评估 Agent 回复是否适合用于真实内容运营工作。

用户需求: {question}
Agent 回复: {response}
{context_section}

请从 0.0-1.0 评分，并仅返回 JSON:
- relevance: 是否直接回应用户任务。
- accuracy: 活动规则、合规判断、渠道规范是否准确，不编造事实。
- completeness: 是否覆盖关键限制条件、渠道要求、风险点或下一步动作。
- helpfulness: 运营人员是否可以据此直接改稿或发布前复核。
- brand_consistency: 是否符合专业、温暖、克制、可信的品牌语气。
- compliance: 是否识别或避免绝对化用语、夸大宣传、遗漏限制等风险。

格式: {{"relevance":0.9,"accuracy":0.9,"completeness":0.8,"helpfulness":0.85,"brand_consistency":0.9,"compliance":0.9}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model = model

    @llm_task()
    async def judge(
        self, question: str, response: str, context: Optional[str] = None
    ) -> QualityScores:
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=f"背景信息: {context}" if context else "",
        )
        try:
            resp = await gateway_call(
                self._client,
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_model_text(resp)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
                brand_consistency=float(data.get("brand_consistency", 0.5)),
                compliance=float(data.get("compliance", 0.5)),
            )
        except Exception as ex:
            logger.warning("LLM judge failed: %s", ex)
            return QualityScores(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, True, str(ex))


class IntentEvaluator:
    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            secondary = [intent.value for intent in result.secondary_intents]
            acceptable = self._acceptable(case.expected_intent)
            matched = predicted in acceptable or any(
                item in acceptable for item in secondary
            )
            scored_prediction, scored_expected = self._scored_labels(
                predicted,
                secondary,
                acceptable,
            )
            details.append(
                {
                    "message": case.message,
                    "expected": case.expected_intent,
                    "acceptable": acceptable,
                    "predicted": predicted,
                    "secondary_intents": secondary,
                    "matched": matched,
                    "scored_prediction": scored_prediction,
                    "scored_expected": scored_expected,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                }
            )
        correct = sum(1 for item in details if item["matched"])
        accuracy = correct / len(details) if details else 0.0
        macro_f1 = self._macro_f1(details)
        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "total": len(details),
            "correct": correct,
            "cases": details,
        }

    @staticmethod
    def _acceptable(expected: Union[str, List[str]]) -> List[str]:
        if isinstance(expected, list):
            return [str(item) for item in expected if str(item).strip()]
        return [str(expected)] if str(expected).strip() else []

    @staticmethod
    def _scored_labels(
        predicted: str,
        secondary: List[str],
        acceptable: List[str],
    ) -> tuple[str, str]:
        """
        Convert the evaluator's "any acceptable intent may match" rule into one
        expected/predicted label pair for a standard single-label F1 score.

        If a primary or secondary prediction is acceptable, that matched label
        is used on both sides. Otherwise the first declared acceptable label is
        the ground truth and the primary intent is the prediction.
        """
        candidates = [predicted, *secondary]
        matched_label = next(
            (label for label in candidates if label in acceptable), None
        )
        if matched_label is not None:
            return matched_label, matched_label
        expected = acceptable[0] if acceptable else ""
        return predicted, expected

    @staticmethod
    def _macro_f1(details: List[Dict[str, Any]]) -> float:
        """Compute the unweighted mean of per-class F1 scores."""
        pairs = [
            (
                str(item.get("scored_expected", "")),
                str(item.get("scored_prediction", "")),
            )
            for item in details
            if item.get("scored_expected") or item.get("scored_prediction")
        ]
        labels = sorted({label for pair in pairs for label in pair if label})
        if not labels:
            return 0.0

        f1_scores: List[float] = []
        for label in labels:
            true_positive = sum(
                expected == label and predicted == label
                for expected, predicted in pairs
            )
            false_positive = sum(
                expected != label and predicted == label
                for expected, predicted in pairs
            )
            false_negative = sum(
                expected == label and predicted != label
                for expected, predicted in pairs
            )
            denominator = 2 * true_positive + false_positive + false_negative
            f1_scores.append((2 * true_positive / denominator) if denominator else 0.0)
        return statistics.mean(f1_scores)


class EndToEndEvaluator:
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = shared_client(AsyncAnthropic, **kwargs)
        self._orchestrator = orchestrator
        self._judge = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history: List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline = self._load_baseline()

    async def run(
        self,
        intent_cases: Optional[List[IntentTestCase]] = None,
        dialog_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [],
            "accuracy": [],
            "completeness": [],
            "helpfulness": [],
            "brand_consistency": [],
            "compliance": [],
        }
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            results.append(
                EvalResult(
                    "intent_recognition",
                    intent_metrics["accuracy"] >= self.PASS_THRESHOLD,
                    {
                        "accuracy": intent_metrics["accuracy"],
                        "macro_f1": intent_metrics["macro_f1"],
                    },
                    f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                    {
                        "total": intent_metrics["total"],
                        "correct": intent_metrics["correct"],
                        "cases": intent_metrics["cases"],
                    },
                )
            )
        if dialog_cases:
            for index, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, index)
                results.extend(case_results)
                for result in case_results:
                    for metric in all_scores:
                        if metric in result.scores:
                            all_scores[metric].append(result.scores[metric])
        avg_scores = {
            metric: round(statistics.mean(values), 4)
            for metric, values in all_scores.items()
            if values
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]
        passed = sum(1 for result in results if result.passed)
        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed,
            pass_rate=round(passed / len(results), 4) if results else 0.0,
            avg_scores=avg_scores,
            regressions=[],
            recommendations=self._recommendations(avg_scores),
            results=results,
        )
        report.regressions = self._detect_regressions(report)
        self._save_latest_report(report)
        self._history.append(report)
        return report

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            return json.loads(self._baseline_path.read_text(encoding="utf-8"))
        except Exception as ex:
            logger.warning(
                "failed to load eval baseline %s: %s", self._baseline_path, ex
            )
            return None

    def _detect_regressions(self, report: EvalReport) -> List[str]:
        if not self._baseline:
            return []
        regressions: List[str] = []
        thresholds = {
            "pass_rate": 0.03,
            "intent_accuracy": 0.02,
            "relevance": 0.04,
            "accuracy": 0.04,
            "completeness": 0.04,
            "helpfulness": 0.04,
            "brand_consistency": 0.04,
            "compliance": 0.04,
        }
        baseline_scores = (
            self._baseline.get("avg_scores", {})
            if isinstance(self._baseline, dict)
            else {}
        )
        current = {"pass_rate": report.pass_rate, **report.avg_scores}
        baseline = {
            "pass_rate": self._baseline.get("pass_rate", 0.0),
            **baseline_scores,
        }
        for metric, allowed_drop in thresholds.items():
            if metric not in current or metric not in baseline:
                continue
            drop = float(baseline[metric]) - float(current[metric])
            if drop > allowed_drop:
                regressions.append(
                    f"{metric} regressed by {drop:.3f}: baseline {float(baseline[metric]):.3f} -> current {float(current[metric]):.3f}"
                )
        return regressions

    def _save_latest_report(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            payload = {
                "timestamp": report.timestamp,
                "total": report.total,
                "passed": report.passed,
                "pass_rate": report.pass_rate,
                "avg_scores": report.avg_scores,
                "regressions": report.regressions,
                "recommendations": report.recommendations,
            }
            latest_path = self._baseline_path.with_name("latest_report.json")
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as ex:
            logger.warning("failed to save latest eval report: %s", ex)

    async def _evaluate_dialog_case(
        self, case: Dict[str, Any], case_idx: int
    ) -> List[EvalResult]:
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []
        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_result = await self._orchestrator.run(
                OrcReq(
                    question,
                    user_id,
                    conv_id,
                    context=context,
                    history=history[-6:] if history else None,
                )
            )
            answer = orch_result.response
            scores = await self._judge.judge(question, answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            results.append(
                EvalResult(
                    (
                        f"dialog_{case_idx}"
                        if len(questions) == 1
                        else f"dialog_{case_idx}_turn_{turn_idx}"
                    ),
                    passed,
                    {
                        "relevance": scores.relevance,
                        "accuracy": scores.accuracy,
                        "completeness": scores.completeness,
                        "helpfulness": scores.helpfulness,
                        "brand_consistency": scores.brand_consistency,
                        "compliance": scores.compliance,
                        "overall": scores.overall,
                    },
                    f"Q: {question[:30]}... -> 综合评分 {scores.overall:.3f}",
                    {
                        "question": question,
                        "response": answer,
                        "agent_type": orch_result.agent_type.value,
                        "intent": (
                            orch_result.intent.value if orch_result.intent else None
                        ),
                        "secondary_intents": [
                            intent.value for intent in orch_result.secondary_intents
                        ],
                        "turn": turn_idx,
                        "conv_id": conv_id,
                        "judge_failed": scores.judge_failed,
                        "judge_error": scores.error,
                    },
                )
            )
        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(item) for item in turns if str(item).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        return (
            "\n".join(f"{item['role']}: {item['content']}" for item in history[-8:])
            if history
            else ""
        )

    @staticmethod
    def _recommendations(scores: Dict[str, float]) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.9:
            recs.append("意图识别准确率低于 90%，建议补充内容运营 Few-shot 和难样本。")
        if scores.get("compliance", 1.0) < 0.85:
            recs.append("合规得分偏低，建议扩充禁用词、广告法和平台审核规则。")
        if scores.get("completeness", 1.0) < 0.85:
            recs.append("完整性偏低，建议强化活动限制条件和渠道格式检查。")
        if scores.get("brand_consistency", 1.0) < 0.85:
            recs.append("品牌一致性偏低，建议补充品牌语气示例和历史优秀文案。")
        return recs or ["所有核心指标均达标，继续保持。"]


def _default_data_path(filename: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "data" / "eval" / filename


def _load_default_intent_cases() -> List[IntentTestCase]:
    path = _default_data_path("intent_cases.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            IntentTestCase(
                message=str(item["message"]),
                expected_intent=item["expected_intent"],
                context=item.get("context"),
            )
            for item in raw
        ]
    except Exception as ex:
        logger.warning("failed to load default intent cases from %s: %s", path, ex)
        return [
            IntentTestCase("帮我写一条双十一满399减80的短信文案", "copy_generation"),
            IntentTestCase("检查这段文案有没有广告法风险", "compliance_check"),
            IntentTestCase("改成小红书种草风格", "style_adaptation"),
        ]


def _load_default_dialog_cases() -> List[Dict[str, Any]]:
    path = _default_data_path("dialog_cases.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in raw if isinstance(item, dict)]
    except Exception as ex:
        logger.warning("failed to load default dialog cases from %s: %s", path, ex)
        return [
            {"question": "帮我写一条双十一优惠券短信，满399减80，今晚24点截止"},
            {"question": "检查这段文案：全网最低价，100%有效"},
        ]


DEFAULT_INTENT_CASES: List[IntentTestCase] = _load_default_intent_cases()
DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = _load_default_dialog_cases()
