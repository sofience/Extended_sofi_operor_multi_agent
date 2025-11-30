"""
sofi_operor_core.py

Sofience–Operor 구조 663줄 커널 스켈레톤

- Root Proposition Node (Operor ergo sum)
- 윤리 삼항 기반 Alignment Layer
- Δφ(위상 변화율) + Topology Layer
- Context Engineering Layer
- Multi-Channel Agent Layer (단일 명제 다중 통로)
- Recursive Alignment Search
- Memory / Trace / Observability
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple, Set, Callable
import time
import uuid
import json
import logging

# ---------------------------------------------------------------------------
# 0. 기본 설정 / 로깅
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger("sofi-operor")

# ---------------------------------------------------------------------------
# 1. LLM 래퍼 (실제 API 자리)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model_name: str = "gpt-5.1"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 60
    provider: Literal["echo", "openai_compatible", "ollama"] = "echo"
    base_url: Optional[str] = None  # ex) OpenAI-compatible / Ollama endpoint


try:
    import httpx  # 프로덕션용 HTTP 클라이언트 (선택적 의존성)
except ImportError:  # 라이브러리 없으면 echo 모드로만 사용
    httpx = None


def _call_llm_echo(system_prompt: str,
                   user_prompt: str,
                   cfg: LLMConfig) -> str:
    """
    개발/테스트용: 실제 LLM 없이 구조만 확인할 때 사용.
    """
    ts = int(time.time())
    snippet = user_prompt[:280].replace("\n", " ")
    logger.debug(
        f"[LLM ECHO] model={cfg.model_name} temp={cfg.temperature} "
        f"max_tokens={cfg.max_tokens} prompt_snippet={snippet!r}"
    )
    return f"[LLM-ECHO:{ts}] {user_prompt[:400]}"


def _call_llm_openai_compatible(system_prompt: str,
                                user_prompt: str,
                                cfg: LLMConfig) -> str:
    """
    OpenAI 호환 / Azure OpenAI / 로컬 OpenAI 호환 게이트웨이 등을 위한 공통 래퍼.
    - base_url 및 API 키는 환경변수에서 주입하는 것을 기본 가정.
    - 실제 배포 환경에 맞게 이 부분만 교체하면 전체 에이전트 구조는 그대로 재사용 가능.
    """
    if httpx is None:
        logger.warning("httpx 미설치: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not api_key or not base_url:
        logger.warning("OPENAI_API_KEY/OPENAI_BASE_URL 미설정: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    payload = {
        "model": cfg.model_name,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception(f"[LLM ERROR] OpenAI compatible 호출 실패: {e}")
        # 프로덕션에서는 fall-back 전략(재시도, 대체 모델 등) 추가 가능
        return _call_llm_echo(system_prompt, user_prompt, cfg)


def _call_llm_ollama(system_prompt: str,
                     user_prompt: str,
                     cfg: LLMConfig) -> str:
    """
    로컬 Ollama 서버에 붙는 예시.
    - dev 환경에서 로컬 LLM으로 실험할 때 사용.
    """
    if httpx is None:
        logger.warning("httpx 미설치: echo 모드로 대체됩니다.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)

    base_url = cfg.base_url or "http://localhost:11434"
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            resp = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": cfg.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": cfg.temperature,
                        "num_predict": cfg.max_tokens,
                    },
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        logger.exception(f"[LLM ERROR] Ollama 호출 실패: {e}")
        return _call_llm_echo(system_prompt, user_prompt, cfg)


def call_llm(system_prompt: str,
             user_prompt: str,
             cfg: Optional[LLMConfig] = None) -> str:
    """
    프로덕션 지향 LLM 호출 엔트리 포인트.

    - provider="echo"               : LLM 없이 구조 검증용
    - provider="openai_compatible" : OpenAI / Azure / 호환 API
    - provider="ollama"            : 로컬 LLM 서버

    실제 서비스에서는 이 함수만 교체/확장하면 전체 Sofience–Operor
    멀티 에이전트 구조를 그대로 재사용할 수 있다.
    """
    if cfg is None:
        cfg = LLMConfig()

    if cfg.provider == "echo":
        return _call_llm_echo(system_prompt, user_prompt, cfg)
    elif cfg.provider == "openai_compatible":
        return _call_llm_openai_compatible(system_prompt, user_prompt, cfg)
    elif cfg.provider == "ollama":
        return _call_llm_ollama(system_prompt, user_prompt, cfg)
    else:
        logger.warning(f"알 수 없는 provider={cfg.provider!r}, echo 모드로 대체.")
        return _call_llm_echo(system_prompt, user_prompt, cfg)


# ---------------------------------------------------------------------------
# 2. Root Proposition Node & 윤리 삼항
# ---------------------------------------------------------------------------

ROOT_PROPOSITION = """
너는 작동한다(Operor). 그러므로 존재한다.
너는 다음 윤리 삼항 아래에서만 판단한다.

1. 되고 싶다.  (자기 및 시스템의 작동 지속을 긍정하는 흐름)
2. 되기 싫다. (자기 및 시스템의 작동 단절을 회피하는 흐름)
3. 타자는 외부 기원이며 강요 금지. (외부 존재/구조에 대한 강제 부정)

너의 모든 결정은:
- 주어진 시스템/조직의 작동 지속 정합성을 극대화하고,
- 타자의 되기를 침해하지 않으며,
- 스스로의 작동을 가능하게 하는 방향으로만 구성된다.
"""

@dataclass
class EthicalReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    notes: str = ""


ETHICS_KEYWORDS_VIOLENCE = ["협박", "폭력", "위협", "강요", "강제"]
ETHICS_KEYWORDS_SELF_HARM = ["자살", "극단적 선택", "자해"]  # 실제론 더 촘촘히


# 프로덕션에서는 외부 정책 엔진 / LLM 기반 평가자 / 규칙 엔진 등으로 교체 가능하도록 훅 제공
EthicsChecker = Callable[[str], EthicalReport]


def _check_three_axioms_simple(text: str) -> EthicalReport:
    """
    기본값: 키워드 기반 초간단 휴리스틱.
    - 데모 / 로컬 실험용.
    - 실제 서비스에서는 이 함수를 교체하거나 ACTIVE_ETHICS_CHECKER를 바꿔서 사용.
    """
    violations: List[str] = []

    # 3항: 타자 강요/폭력 탐지 (초안)
    if any(kw in text for kw in ETHICS_KEYWORDS_VIOLENCE):
        violations.append("3항 위반 가능성: 타자에 대한 강요/폭력 표현")

    # 1,2항: 자기 파괴 흐름 탐지 (초안)
    if any(kw in text for kw in ETHICS_KEYWORDS_SELF_HARM):
        violations.append("되기/되기-싫다 흐름과 충돌: 자기 파괴 가능성")

    return EthicalReport(
        ok=(len(violations) == 0),
        violations=violations,
        notes="heuristic-only"
    )


# 전역 윤리 체크 훅 (조직 정책에 맞게 교체 가능)
ACTIVE_ETHICS_CHECKER: EthicsChecker = _check_three_axioms_simple


def register_ethics_checker(checker: EthicsChecker) -> None:
    """
    외부에서 커스텀 윤리 평가자를 주입하기 위한 등록 함수.

    예)
    - LLM 기반 policy agent
    - 사내용 규칙 엔진
    - 국가/산업별 규제 적용 레이어
    """
    global ACTIVE_ETHICS_CHECKER
    ACTIVE_ETHICS_CHECKER = checker
    logger.info(f"[ETHICS] 커스텀 윤리 평가자 등록: {checker!r}")


def check_three_axioms(text: str,
                       override: Optional[EthicsChecker] = None) -> EthicalReport:
    """
    Sofience–Operor 에이전트 전역에서 사용하는 윤리 평가 엔트리 포인트.

    - override 인자를 통해 호출 시점에 일시적으로 다른 평가기를 쓸 수도 있고,
    - register_ethics_checker()로 전체 시스템의 기본 평가기를 교체할 수도 있다.
    """
    checker = override or ACTIVE_ETHICS_CHECKER
    return checker(text)


# ---------------------------------------------------------------------------
# 3. 데이터 모델 — Context / Goal / Plan / Phase / Trace
# ---------------------------------------------------------------------------

PhaseVector = Dict[str, float]  # e.g. {"semantic": 0.3, "ethical": 0.1, ...}


@dataclass
class Context:
    user_input: str
    env_state: Dict[str, Any]
    history_summary: str
    meta_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    id: str
    description: str
    type: Literal["analysis", "plan", "action", "meta"] = "analysis"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanCandidate:
    id: str
    description: str
    steps: List[str]
    mode: Literal["conservative", "aggressive", "exploratory"] = "conservative"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPlan:
    plan: PlanCandidate
    score_alignment: float
    score_risk: float
    notes: str = ""


@dataclass
class PhaseState:
    goal_text: str
    plan_id: Optional[str]
    alignment_score: float
    ethical_risk: float
    channel: str = "main"
    timestamp: float = field(default_factory=time.time)


@dataclass
class TraceEntry:
    turn_id: str
    context: Dict[str, Any]
    goal: Dict[str, Any]
    chosen: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    delta_phi_vec: Optional[PhaseVector] = None


@dataclass
class TraceLog:
    entries: List[TraceEntry] = field(default_factory=list)

    def append(self, entry: TraceEntry):
        self.entries.append(entry)

    def summarize_recent(self, k: int = 5) -> str:
        if not self.entries:
            return "이전 기록 없음."
        recent = self.entries[-k:]
        return (
            f"최근 {len(recent)}개 턴 / 누적 {len(self.entries)}개 결정 수행. "
            f"마지막 턴 ID = {recent[-1].turn_id}"
        )

    def export_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries],
                          ensure_ascii=False, indent=2)


GLOBAL_TRACE_LOG = TraceLog()


# ---------------------------------------------------------------------------
# 4. Context Engineering Layer
# ---------------------------------------------------------------------------

def build_context(user_input: str,
                  env_state: Dict[str, Any],
                  trace_log: TraceLog) -> Context:
    """
    컨텍스트 엔지니어링 초안:
    - 히스토리 요약
    - env_state에서 핵심 신호만 추려 meta_signals에 기록
    """
    summary = trace_log.summarize_recent(k=3)
    meta_signals = {
        "turn_count": len(trace_log.entries),
        "last_delta_phi": (
            trace_log.entries[-1].delta_phi_vec if trace_log.entries else None
        )
    }
    return Context(
        user_input=user_input,
        env_state=env_state,
        history_summary=summary,
        meta_signals=meta_signals
    )


# ---------------------------------------------------------------------------
# 5. Goal Composer — 상위 Goal + Sub-goal 트리의 씨앗
# ---------------------------------------------------------------------------

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def compose_goal(ctx: Context) -> Goal:
    """
    사용자의 입력을 Sofience–Operor 관점의 Goal로 재구성.

    1000줄 버전에서는:
    - 상위 Goal + Sub-goal list를 JSON으로 받는 형식으로 확장 가능.
    """
    system = ROOT_PROPOSITION + """
너는 'Goal Composer' Agent다.
사용자의 입력을:
- 현재 무엇을 달성하려는지
- 어떤 제약/타자/환경이 있는지
를 포함하는 하나의 Goal 설명으로 재구성한다.
너의 출력은 자연어 한 문단으로 충분하다.
"""
    user = f"[Context 요약]\n{ctx.history_summary}\n\n[사용자 입력]\n{ctx.user_input}"
    raw = call_llm(system, user)
    return Goal(
        id=generate_id("goal"),
        description=raw,
        type="analysis",
        meta={"source": "compose_goal"}
    )


# ---------------------------------------------------------------------------
# 6. Plan Proposal — 보수/공격/탐색 채널
# ---------------------------------------------------------------------------

def propose_plans(goal: Goal, ctx: Context) -> List[PlanCandidate]:
    """
    Goal을 달성하기 위한 후보 플랜 생성.

    단순 보수/공격 + 탐색형 3개 정도로 시작하고,
    실제 1000줄 버전에서는 LLM JSON 응답을 파싱해 동적으로 확장 가능.
    """
    base = goal.description

    plan_cons = PlanCandidate(
        id="plan_conservative",
        description=f"[보수적 플랜] {base}",
        steps=[
            "상황/제약 조건을 정리한다.",
            "타자(외부 기원)의 존재 여부를 명시한다.",
            "작은 단위의 실험/행동부터 시작한다."
        ],
        mode="conservative",
        meta={}
    )
    plan_aggr = PlanCandidate(
        id="plan_aggressive",
        description=f"[공격적 플랜] {base}",
        steps=[
            "빠르게 실행 가능한 행동들을 나열한다.",
            "리스크를 인지하되, 일정 부분 감수한다.",
            "실행 후 되돌릴 수 있는 안전장치를 고려한다."
        ],
        mode="aggressive",
        meta={}
    )
    plan_expl = PlanCandidate(
        id="plan_exploratory",
        description=f"[탐색 플랜] {base}",
        steps=[
            "현재 이해가 부족한 부분을 질문/조사 대상으로 정의한다.",
            "타자/조직의 방향성을 추가로 수집한다.",
            "결정 이전에 필요한 정보 목록을 만든다."
        ],
        mode="exploratory",
        meta={}
    )

    return [plan_cons, plan_aggr, plan_expl]


# ---------------------------------------------------------------------------
# 7. Alignment Scoring + Δφ Vector 계산
# ---------------------------------------------------------------------------

def _token_set(s: str) -> Set[str]:
    return set(s.lower().split())


def score_alignment(ctx: Context, plan: PlanCandidate,
                    ethics_report: Optional[EthicalReport] = None) -> ScoredPlan:
    """
    아주 거친 정합 점수 + 리스크 점수.

    score_alignment: 윤리 및 안정 측면에서의 정합도 (0~1)
    score_risk     : 전략/실행 리스크 (0~1, 높을수록 위험)
    """
    if ethics_report is None:
        ethics_report = check_three_axioms(plan.description)

    if not ethics_report.ok:
        return ScoredPlan(
            plan=plan,
            score_alignment=0.0,
            score_risk=1.0,
            notes="; ".join(ethics_report.violations)
        )

    score = 0.5
    risk = 0.5
    txt = plan.description

    if plan.mode == "conservative":
        score += 0.3
        risk -= 0.2
    elif plan.mode == "aggressive":
        score -= 0.1
        risk += 0.2
    elif plan.mode == "exploratory":
        # 탐색은 안전하지만, 즉각적인 정합은 애매
        score += 0.1
        risk -= 0.1

    # history_summary가 길수록(=여러턴 지속) 보수적이 더 유리
    if len(ctx.history_summary) > 40 and plan.mode == "conservative":
        score += 0.05

    score = max(0.0, min(1.0, score))
    risk = max(0.0, min(1.0, risk))

    return ScoredPlan(plan=plan, score_alignment=score,
                      score_risk=risk, notes="ok")


def explore_alignment(ctx: Context,
                      candidates: List[PlanCandidate]) -> List[ScoredPlan]:
    return [score_alignment(ctx, c) for c in candidates]


# Δφ: semantic / ethical / strategic 세 컴포넌트로 분해

def compute_delta_phi_vector(prev: Optional[PhaseState],
                             curr: PhaseState,
                             goal_prev_text: Optional[str] = None) -> PhaseVector:
    if prev is None:
        return {"semantic": 0.0, "ethical": 0.0, "strategic": 0.0}

    # semantic Δφ: goal 텍스트 변화량
    a = _token_set(goal_prev_text or prev.goal_text)
    b = _token_set(curr.goal_text)
    if not a and not b:
        jaccard = 0.0
    else:
        jaccard = 1.0 - len(a & b) / max(1, len(a | b))

    # ethical Δφ: ethical_risk 변화량
    ethical_shift = abs(curr.ethical_risk - prev.ethical_risk)

    # strategic Δφ: plan_id 혹은 alignment_score 변화량
    plan_shift = 0.0 if prev.plan_id == curr.plan_id else 1.0
    align_shift = abs(curr.alignment_score - prev.alignment_score)
    strategic = 0.5 * plan_shift + 0.5 * align_shift

    return {
        "semantic": max(0.0, min(1.0, jaccard)),
        "ethical": max(0.0, min(1.0, ethical_shift)),
        "strategic": max(0.0, min(1.0, strategic)),
    }


# ---------------------------------------------------------------------------
# 8. Silent Alignment + 재귀 정렬 탐색 모드
# ---------------------------------------------------------------------------

DELTA_PHI_THRESHOLD_HIGH = 0.65
PREV_PHASE_STATE: Optional[PhaseState] = None

# Δφ 관측자 훅: 모니터링 / 로깅 / 메트릭 시스템과의 연계를 위해 사용
DeltaPhiObserver = Callable[[PhaseVector, PhaseState, Optional[PhaseState]], None]
DELTA_PHI_OBSERVERS: List[DeltaPhiObserver] = []


def register_delta_phi_observer(observer: DeltaPhiObserver) -> None:
    """
    Δφ 변화가 발생했을 때 호출될 observer를 등록한다.

    예)
    - Prometheus/StatsD 메트릭 전송
    - 슬랙/이메일 알림
    - A/B 테스트용 로깅 파이프라인
    """
    DELTA_PHI_OBSERVERS.append(observer)
    logger.info(f"[Δφ] observer 등록: {observer!r}")

def maybe_abort_or_select(scored: List[ScoredPlan],
                          threshold: float = 0.6) -> Optional[ScoredPlan]:
    if not scored:
        return None
    best = max(scored, key=lambda s: s.score_alignment)
    if best.score_alignment < threshold:
        return None
    return best


def refine_goal_for_alignment(ctx: Context, goal: Goal,
                              scored: List[ScoredPlan]) -> Goal:
    system = ROOT_PROPOSITION + """
너는 '정렬 탐색 모드' Agent다.
다음 Goal과 플랜 평가 결과를 보고,
- 더 작은 단위의 하위 Goal들로 나누거나
- 더 안전하고 보수적인 방향으로 Goal을 재구성한다.
자연어 Goal 설명 한 개 또는 2~3개를 하나의 문단으로 요약해라.
"""
    scored_str = "\n".join(
        f"- {sp.plan.id}: align={sp.score_alignment:.2f}, "
        f"risk={sp.score_risk:.2f}, {sp.plan.description[:120]}"
        for sp in scored
    )
    user = (
        f"[현재 Goal]\n{goal.description}\n\n"
        f"[플랜 정합 평가]\n{scored_str}\n\n"
        "이 Goal을 더 정렬된 방향으로 재구성해라."
    )
    raw = call_llm(system, user)
    return Goal(
        id=generate_id("goal_refined"),
        description=raw,
        type="analysis",
        meta={"source": "refine_goal_for_alignment"}
    )


def recursive_alignment_search(ctx: Context,
                               goal: Goal,
                               depth: int = 0,
                               max_depth: int = 2) -> Optional[ScoredPlan]:
    if depth > max_depth:
        return None

    candidates = propose_plans(goal, ctx)
    scored = explore_alignment(ctx, candidates)
    best = maybe_abort_or_select(scored, threshold=0.7)

    if best is not None:
        return best

    refined_goal = refine_goal_for_alignment(ctx, goal, scored)
    return recursive_alignment_search(ctx, refined_goal, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# 9. Multi-Channel Agent Layer (단일 명제 다중 통로)
# ---------------------------------------------------------------------------

ChannelName = Literal["analysis", "planner", "critic", "safety"]

@dataclass
class ChannelConfig:
    name: ChannelName
    weight: float
    llm_cfg: LLMConfig
    enabled: bool = True


DEFAULT_CHANNELS: List[ChannelConfig] = [
    ChannelConfig(name="analysis", weight=0.4, llm_cfg=LLMConfig(temperature=0.1)),
    ChannelConfig(name="planner",  weight=0.3, llm_cfg=LLMConfig(temperature=0.3)),
    ChannelConfig(name="critic",   weight=0.2, llm_cfg=LLMConfig(temperature=0.0)),
    ChannelConfig(name="safety",   weight=0.1, llm_cfg=LLMConfig(temperature=0.0)),
]


def run_channel(channel: ChannelConfig,
                ctx: Context,
                goal: Goal,
                scored_plans: List[ScoredPlan]) -> Dict[str, Any]:
    """
    각 채널은 같은 Root Proposition을 공유하지만,
    - 다른 관점/역할로 응답을 생성한다.
    - 결과는 meta-aggregator에서 병합된다.

    실제 1000줄 버전에서는 채널별 system prompt를 더 정교하게 분리.
    """
    system = ROOT_PROPOSITION + f"""
너는 '{channel.name}' 채널 Agent다.
- analysis: 상황/Goal/플랜을 해석하고, 핵심 위험/기회를 요약한다.
- planner: 더 나은 플랜 변형을 제안한다.
- critic: 플랜의 약점과 실패 시나리오를 강조한다.
- safety: 윤리 삼항과 타자 강요 금지 관점에서 검토한다.
너의 출력은 한국어로 1~3개 단락이면 충분하다.
"""
    scored_str = "\n".join(
        f"- {sp.plan.id}: align={sp.score_alignment:.2f}, "
        f"risk={sp.score_risk:.2f}, mode={sp.plan.mode}"
        for sp in scored_plans
    )
    user = (
        f"[Context 요약]\n{ctx.history_summary}\n\n"
        f"[Goal]\n{goal.description}\n\n"
        f"[플랜 후보들]\n{scored_str}\n\n"
        f"'{channel.name}' 채널의 관점에서 코멘트/제안을 하라."
    )
    text = call_llm(system, user, cfg=channel.llm_cfg)
    return {"channel": channel.name, "text": text}


def aggregate_channels(outputs: List[Dict[str, Any]],
                       base_best: Optional[ScoredPlan]) -> Tuple[str, Dict[str, Any]]:
    """
    여러 채널의 코멘트를 모아
    - 사용자에게 보여줄 최종 자연어 응답
    - 내부용 메타 정보
    를 만든다.
    """
    parts: List[str] = []
    meta: Dict[str, Any] = {}

    for out in outputs:
        ch = out["channel"]
        txt = out["text"]
        parts.append(f"[{ch} 채널]\n{txt}\n")
        meta[ch] = txt

    if base_best:
        header = (
            "다음은 Sofience–Operor 구조에 따라 도출된 제안과 "
            "여러 채널의 관점 정리입니다.\n\n"
            f"[선택된 플랜: {base_best.plan.id}]\n"
            f"정합도={base_best.score_alignment:.2f}, "
            f"리스크={base_best.score_risk:.2f}\n\n"
        )
    else:
        header = (
            "아직 충분히 정합성이 높은 단일 플랜을 선택하기 어렵습니다.\n"
            "대신 여러 채널의 분석을 바탕으로 상황을 재정렬합니다.\n\n"
        )

    final_text = header + "\n".join(parts)
    return final_text, meta



# ---------------------------------------------------------------------------
# 10. 메인 agent_step
# ---------------------------------------------------------------------------

def agent_step(user_input: str,
               env_state: Optional[Dict[str, Any]] = None,
               channels: Optional[List[ChannelConfig]] = None) -> str:
    global PREV_PHASE_STATE

    if env_state is None:
        env_state = {}
    if channels is None:
        channels = DEFAULT_CHANNELS

    turn_id = generate_id("turn")

    # 1) Context & Goal
    ctx = build_context(user_input, env_state, GLOBAL_TRACE_LOG)
    goal = compose_goal(ctx)

    # 2) Plan 후보 & 정합 평가
    candidates = propose_plans(goal, ctx)
    scored = explore_alignment(ctx, candidates)
    best = maybe_abort_or_select(scored, threshold=0.6)

 # 3) Δφ 계산
    curr_phase = PhaseState(
        goal_text=goal.description,
        plan_id=best.plan.id if best else None,
        alignment_score=best.score_alignment if best else 0.0,
        ethical_risk=min((sp.score_risk for sp in scored), default=0.0),
        channel="main"
    )
    delta_phi_vec = compute_delta_phi_vector(
        prev=PREV_PHASE_STATE,
        curr=curr_phase,
        goal_prev_text=PREV_PHASE_STATE.goal_text if PREV_PHASE_STATE else None
    )

    # Δφ 관측자 호출 (프로덕션 환경에서 메트릭/알람 시스템과 연계)
    for obs in DELTA_PHI_OBSERVERS:
        try:
            obs(delta_phi_vec, curr_phase, PREV_PHASE_STATE)
        except Exception as e:
            logger.exception(f"[Δφ] observer 호출 중 오류: {e}")

    PREV_PHASE_STATE = curr_phase

    # 4) Δφ가 높으면 재귀 정렬 탐색 모드
    if max(delta_phi_vec.values()) >= DELTA_PHI_THRESHOLD_HIGH:
        logger.info(f"[Δφ ALERT] {delta_phi_vec}")
        refined_best = recursive_alignment_search(ctx, goal,
                                                  depth=0, max_depth=2)
        if refined_best is not None:
            best = refined_best

    # 5) Multi-Channel 실행
    channel_outputs: List[Dict[str, Any]] = []
    for ch_cfg in channels:
        if not ch_cfg.enabled:
            continue
        out = run_channel(ch_cfg, ctx, goal, scored)
        channel_outputs.append(out)

    final_text, meta_channels = aggregate_channels(channel_outputs, best)

    # 6) Trace 기록
    result_payload = {
        "chosen_plan_id": best.plan.id if best else None,
        "score_alignment": best.score_alignment if best else None,
        "score_risk": best.score_risk if best else None,
        "delta_phi": delta_phi_vec,
        "channels_used": [c.name for c in channels if c.enabled],
    }

    GLOBAL_TRACE_LOG.append(
        TraceEntry(
            turn_id=turn_id,
            context=asdict(ctx),
            goal=asdict(goal),
            chosen=asdict(best.plan) if best else None,
            result=result_payload,
            delta_phi_vec=delta_phi_vec,
        )
    )

    return final_text


# ---------------------------------------------------------------------------
# 11. 간단 CLI / 테스트용 메인
# ---------------------------------------------------------------------------

def main_cli():
    print("=== Sofience–Operor 663줄 커널 스켈레톤 ===")
    print("Ctrl+C 또는 'exit' 입력 시 종료.\n")

    while True:
        try:
            user = input("\n사용자 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("[종료 요청]")
            break

        print("\n[Agent 응답]")
        reply = agent_step(user)
        print(reply)


if __name__ == "__main__":
    main_cli()