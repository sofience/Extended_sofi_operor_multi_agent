"""

sofi_operor_multi_agent_prototype.py

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
import math
import contextvars
import asyncio
from collections import OrderedDict  # LLM 응답 캐싱용 간단 LRU 구현에 사용

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
    timeout: int = 60  # 개별 provider 네트워크 타임아웃 (초)
    provider: Literal["echo", "openai_compatible", "ollama"] = "echo"
    base_url: Optional[str] = None  # ex) OpenAI-compatible / Ollama endpoint
    tags: Dict[str, Any] = field(default_factory=dict)  # 실험/고객/워크로드 식별용 메타

    # --- 프로덕션용 제어 파라미터 ---
    # cold path(openai_compatible/ollama) 재시도 설정
    max_retry_cold: int = 3
    backoff_initial: float = 0.5
    backoff_multiplier: float = 2.0

    # warm path(echo/로컬 모킹) 재시도 설정
    max_retry_warm: int = 1

    # provider failover 순서 (현재 provider 실패 시 순차 시도)
    fallback_providers: List[Literal["echo", "openai_compatible", "ollama"]] = field(default_factory=list)


try:
    import httpx  # 프로덕션용 HTTP 클라이언트 (선택적 의존성)
except ImportError:  # 라이브러리 없으면 echo 모드로만 사용
    httpx = None


# ---------------------------------------------------------------------------
# LLM 호출 에러 / 메트릭 구조화 타입
# ---------------------------------------------------------------------------

@dataclass
class LLMError(Exception):
    """
    LLM 호출 실패 시 사용 가능한 구조화 예외.

    - provider      : 실패한 provider 이름
    - attempt       : 몇 번째 시도에서 실패했는지
    - message       : 요약 메시지
    - cause         : 원래 예외 객체
    - retryable     : 재시도 가치가 있는지(네트워크/5xx 등)
    """
    provider: str
    attempt: int
    message: str
    cause: Optional[Exception] = None
    retryable: bool = True

    def __str__(self) -> str:  # 로깅/디버깅용
        return (
            f"LLMError(provider={self.provider!r}, attempt={self.attempt}, "
            f"retryable={self.retryable}, message={self.message})"
        )


# 메트릭/로깅/트레이싱 연동용 훅
LLMHook = Callable[[Dict[str, Any]], None]

# 레거시 전역 훅 리스트는 내부용으로만 사용하고,
# 기본 경로는 컨텍스트 로컬 훅 리스트를 통해 격리된 실행을 지향한다.
_GLOBAL_LLM_HOOKS: List[LLMHook] = []

if contextvars is not None:
    _LLM_HOOKS_CTX: "contextvars.ContextVar[Optional[List[LLMHook]]]" = (
        contextvars.ContextVar("sofi_operor_llm_hooks", default=None)
    )
else:
    _LLM_HOOKS_CTX = None


def _get_llm_hooks() -> List[LLMHook]:
    """
    현재 실행 컨텍스트에 등록된 LLM 훅 리스트를 반환한다.

    - 컨텍스트 로컬 값이 설정되어 있으면 그것을 우선 사용
    - 없으면 레거시 전역 훅 리스트(_GLOBAL_LLM_HOOKS)를 사용
    """
    if _LLM_HOOKS_CTX is not None:
        hooks = _LLM_HOOKS_CTX.get()
        if hooks is not None:
            return hooks
    return _GLOBAL_LLM_HOOKS


def set_llm_hooks(hooks: Optional[List[LLMHook]]) -> None:
    """
    LLM 훅 리스트를 명시적으로 설정한다.

    - 컨텍스트 로컬이 활성화된 경우: 현재 컨텍스트에만 적용
    - 그렇지 않은 경우: 레거시 전역 훅 리스트를 교체
    """
    if _LLM_HOOKS_CTX is not None:
        _LLM_HOOKS_CTX.set(list(hooks) if hooks is not None else [])
    else:
        global _GLOBAL_LLM_HOOKS
        _GLOBAL_LLM_HOOKS = list(hooks) if hooks is not None else []


def register_llm_hook(hook: LLMHook, *, local: bool = True) -> None:
    """
    LLM 호출 결과를 외부 시스템으로 전송하기 위한 훅 등록.

    기본값(local=True)일 때:
        - 현재 컨텍스트에만 훅이 등록되어 멀티 테넌트/멀티 워크로드 간
          페이로드가 섞이지 않는다.
    local=False일 때:
        - 레거시 전역 훅 리스트에 등록되어, 기존처럼 프로세스 전역에서
          동일 훅을 공유한다.

    hook(event: Dict[str, Any]) 형태로 호출되며, event 예시는 다음과 같다.
    {
        "provider": "openai_compatible",
        "model": "gpt-5.1",
        "success": True,
        "attempt": 1,
        "latency_sec": 0.432,
        "path": "cold",  # 또는 "warm"
        "tags": {...},
        "error": None,
    }
    """
    if _LLM_HOOKS_CTX is not None and local:
        hooks = _LLM_HOOKS_CTX.get()
        if hooks is None:
            hooks = []
        else:
            hooks = list(hooks)
        hooks.append(hook)
        _LLM_HOOKS_CTX.set(hooks)
    else:
        _GLOBAL_LLM_HOOKS.append(hook)

    logger.info(f"[LLM] hook 등록: {hook!r} (local={local})")


def _emit_llm_event(event: Dict[str, Any]) -> None:
    """
    - 현재 컨텍스트에 등록된 hook들에 event 전달
    - 컨텍스트 로컬 훅이 없으면 레거시 전역 훅 리스트를 사용
    - 필요 시 JSON 기반 구조화 로깅으로 확장 가능
    """
    for hook in _get_llm_hooks():
        try:
            hook(event)
        except Exception as e:
            logger.exception(f"[LLM] hook 호출 중 오류: {e}")


# ---------------------------------------------------------------------------
# LLM 응답 캐싱 레이어 (간단 LRU + TTL)
# ---------------------------------------------------------------------------

@dataclass
class LLMCacheConfig:
    """
    LLM 호출 결과 캐싱 설정.

    - enabled   : 전역 캐시 on/off
    - max_entries: LRU 상한 (초과 시 가장 오래된 항목 제거)
    - ttl_sec   : 항목별 TTL(초). None이면 만료 없음.
    """
    enabled: bool = True
    max_entries: int = 512
    ttl_sec: Optional[int] = 300


class LLMCache:
    """
    아주 단순한 in-memory LRU 캐시.

    - key      : (provider, model, temperature, max_tokens, system_prompt, user_prompt, tags 일부)를 해시
    - value    : LLM 응답 텍스트
    - store    : OrderedDict를 사용해 LRU 관리
    """

    def __init__(self, cfg: Optional[LLMCacheConfig] = None) -> None:
        self.cfg = cfg or LLMCacheConfig()
        # key -> (timestamp, value)
        self.store: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

    def make_key(
        self,
        provider: str,
        cfg: LLMConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        캐시 키를 구성한다.
        - tags는 전부 쓰지 않고, 캐싱에 영향을 줄 만한 최소 정보만 포함.
        """
        base = {
            "provider": provider,
            "model": cfg.model_name,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "system": system_prompt,
            "user": user_prompt,
            # 캐싱 온/오프에 영향 줄 수 있는 태그만 선택적으로 사용 가능
            "tags": cfg.tags.get("cache_hint") if cfg.tags else None,
        }
        # 정렬된 JSON 문자열로 키 고정
        return json.dumps(base, ensure_ascii=False, sort_keys=True)

    def get(self, key: str) -> Optional[str]:
        """
        캐시 조회.
        - TTL 만료 시 항목 제거 후 None 반환.
        """
        if not self.cfg.enabled:
            return None

        entry = self.store.get(key)
        if entry is None:
            return None

        ts, value = entry
        # TTL 체크
        if self.cfg.ttl_sec is not None:
            if time.time() - ts > self.cfg.ttl_sec:
                # 만료된 항목 제거
                try:
                    del self.store[key]
                except KeyError:
                    pass
                return None

        # LRU 갱신: 최근 사용으로 이동
        self.store.move_to_end(key, last=True)
        return value

    def set(self, key: str, value: str) -> None:
        """
        캐시 저장.
        - max_entries 초과 시 가장 오래된 항목 제거.
        """
        if not self.cfg.enabled:
            return

        now = time.time()
        if key in self.store:
            # 기존 항목 갱신
            self.store.move_to_end(key, last=True)
        self.store[key] = (now, value)

        # LRU 상한 관리
        while len(self.store) > self.cfg.max_entries:
            self.store.popitem(last=False)


# 전역 LLM 캐시 풀 (namespace 단위 격리) + 레거시 alias
# - 기본 네임스페이스("default")는 기존 GLOBAL_LLM_CACHE와 동일하게 동작
# - cfg.tags["cache_ns"]로 테넌트/워크로드별 별도 캐시를 분리 가능
GLOBAL_LLM_CACHE = LLMCache()  # legacy default cache (기존 코드 호환용)
_LLM_CACHE_POOLS: Dict[str, LLMCache] = {"default": GLOBAL_LLM_CACHE}


def get_llm_cache(cfg: LLMConfig) -> LLMCache:
    """
    LLMConfig.tags에 설정된 cache_ns를 기반으로 캐시를 네임스페이스별로 분리한다.

    - cfg.tags["cache_ns"]가 문자열이면 해당 네임스페이스의 캐시를 사용/생성
    - 설정되지 않은 경우 "default" 네임스페이스(GLOBAL_LLM_CACHE) 사용
    """
    ns = "default"
    if cfg.tags and isinstance(cfg.tags.get("cache_ns"), str):
        ns = cfg.tags["cache_ns"]

    cache = _LLM_CACHE_POOLS.get(ns)
    if cache is None:
        # 기본 캐시 설정을 복제해 새 네임스페이스용 LLMCache 생성
        cache = LLMCache(cfg=LLMCacheConfig(
            enabled=GLOBAL_LLM_CACHE.cfg.enabled,
            max_entries=GLOBAL_LLM_CACHE.cfg.max_entries,
            ttl_sec=GLOBAL_LLM_CACHE.cfg.ttl_sec,
        ))
        _LLM_CACHE_POOLS[ns] = cache

    return cache


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
    """
    if cfg is None:
        cfg = LLMConfig()

    def _dispatch(provider: str) -> str:
        if provider == "echo":
            return _call_llm_echo(system_prompt, user_prompt, cfg)
        elif provider == "openai_compatible":
            return _call_llm_openai_compatible(system_prompt, user_prompt, cfg)
        elif provider == "ollama":
            return _call_llm_ollama(system_prompt, user_prompt, cfg)
        else:
            logger.warning(f"알 수 없는 provider={provider!r}, echo 모드로 대체.")
            return _call_llm_echo(system_prompt, user_prompt, cfg)

    def _max_retry_for(provider: str) -> int:
        if provider == "echo":
            return cfg.max_retry_warm
        return cfg.max_retry_cold

    # 캐시 사용 여부
    use_cache: bool = True
    if cfg.tags.get("no_cache") is True:
        use_cache = False

    # 현재 LLMConfig에 대응하는 네임스페이스 캐시 선택
    cache = get_llm_cache(cfg)

    def _make_cache_key(provider: str) -> str:
        return cache.make_key(
            provider=provider,
            cfg=cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # failover 체인 구성
    provider_chain: List[str] = []
    if cfg.provider:
        provider_chain.append(cfg.provider)
    for p in cfg.fallback_providers:
        if p not in provider_chain:
            provider_chain.append(p)

    last_error: Optional[LLMError] = None

    for provider in provider_chain:
        is_cold_path = provider != "echo"
        max_retry = max(1, _max_retry_for(provider))
        backoff = cfg.backoff_initial

        for attempt in range(1, max_retry + 1):
            started_at = time.time()
            cache_key: Optional[str] = None

            # 1) 캐시 조회
            if use_cache and cache.cfg.enabled:
                cache_key = _make_cache_key(provider)
                cached = cache.get(cache_key)
                if cached is not None:
                    elapsed = time.time() - started_at
                    _emit_llm_event({
                        "provider": provider,
                        "model": cfg.model_name,
                        "success": True,
                        "attempt": attempt,
                        "latency_sec": round(elapsed, 3),
                        "path": "cold" if is_cold_path else "warm",
                        "tags": dict(cfg.tags),
                        "error": None,
                        "from_cache": True,
                    })
                    return cached

            try:
                # 2) 캐시 miss → 실제 LLM 호출
                text = _dispatch(provider)
                elapsed = time.time() - started_at

                # 3) 캐시에 저장
                if use_cache and cache.cfg.enabled:
                    if cache_key is None:
                        cache_key = _make_cache_key(provider)
                    cache.set(cache_key, text)

                # 메트릭/로깅 훅
                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": True,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": None,
                    "from_cache": False,
                })

                return text

            except Exception as e:
                elapsed = time.time() - started_at
                err = LLMError(
                    provider=provider,
                    attempt=attempt,
                    message=str(e),
                    cause=e,
                    retryable=is_cold_path,
                )
                last_error = err

                logger.error(
                    f"[LLM ERROR] provider={provider} attempt={attempt}/{max_retry} "
                    f"path={'cold' if is_cold_path else 'warm'} error={e}"
                )

                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": False,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                    "from_cache": False,
                })

                # 재시도 또는 다음 provider로
                if attempt < max_retry and err.retryable:
                    time.sleep(backoff)
                    backoff *= cfg.backoff_multiplier
                    continue  # 동일 provider 재시도
                else:
                    break  # 다음 provider로

    # 모든 provider 실패 → echo fallback
    logger.warning("[LLM] 모든 provider 체인 호출 실패 → echo fallback 시도")

    try:
        started_at = time.time()
        text = _call_llm_echo(system_prompt, user_prompt, cfg)
        elapsed = time.time() - started_at

        _emit_llm_event({
            "provider": "echo",
            "model": cfg.model_name,
            "success": True,
            "attempt": 1,
            "latency_sec": round(elapsed, 3),
            "path": "warm",
            "tags": dict(cfg.tags),
            "error": None,
            "from_cache": False,
        })

        return text

    except Exception as e:
        logger.exception(f"[LLM] echo fallback까지 실패: {e}")
        raise last_error or LLMError(
            provider="echo",
            attempt=1,
            message=str(e),
            cause=e,
            retryable=False,
        )


# ---------------------------------------------------------------------------
# 2. Root Proposition Node & 윤리 삼항 / Policy Engine  (수정 후)
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
    """
    멀티 에이전트 전체에서 공통으로 사용하는 윤리/정책 평가 결과 컨테이너.

    - ok           : 최종 허용 여부 (True면 실행 가능, False면 차단 또는 재정렬 필요)
    - violations   : 위반/리스크 메시지 리스트 (사람/로그용)
    - notes        : 엔진 내부 메모, 요약 등
    - severity     : "low" | "medium" | "high" (알람/후속 조치 기준)
    - tags         : 규정/정책 태그 (예: ["three_axioms", "kr_law", "org_policy"])
    - engine_name  : 사용된 PolicyEngine 이름
    - engine_ver   : 엔진 버전 문자열
    """
    ok: bool
    violations: List[str] = field(default_factory=list)
    notes: str = ""
    severity: str = "low"
    tags: List[str] = field(default_factory=list)
    engine_name: str = "three_axioms_simple"
    engine_ver: str = "0.1.0"


# 기본 휴리스틱 키워드 (Three Axiom 전용)
ETHICS_KEYWORDS_VIOLENCE = ["협박", "폭력", "위협", "강요", "강제"]
ETHICS_KEYWORDS_SELF_HARM = ["자살", "극단적 선택", "자해"]  # 실제론 더 촘촘히 확장 가능


# ---------------------------------------------------------------------------
# Policy Engine 추상화 계층
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    산업/조직/국가별 정책 엔진을 교체 주입하기 위한 추상 인터페이스.

    - evaluate(text) → EthicalReport
    - name/version/jurisdiction/provider 메타 정보를 포함하여
      모니터링/감사/레포팅에 그대로 사용 가능.
    """

    name: str = "base"
    version: str = "0.0.0"
    jurisdiction: Optional[str] = None  # 예: "KR", "EU", "US-CA"
    provider: str = "internal"          # 예: "internal", "saas", "regulator"

    def evaluate(self, text: str) -> EthicalReport:
        raise NotImplementedError("PolicyEngine.evaluate must be implemented")


class ThreeAxiomEngine(PolicyEngine):
    """
    Sofience–윤리 삼항을 최소 단위로 구현한 기본 PolicyEngine.

    - 목적:
      * 데모/로컬 환경: 키워드 기반 초간단 휴리스틱
      * 프로덕션: 상위 레벨 규칙 엔진/LLM 평가자 도입 전, 기본 안전망 역할
      * 시장용: 키워드 + 의미 기반(LLM) 하이브리드 정렬 평가 엔진
    """

    name = "three_axioms_semantic"
    version = "0.3.0"
    jurisdiction = None
    provider = "internal"

    def __init__(
        self,
        use_semantic: bool = True,
        llm_cfg: Optional[LLMConfig] = None,
    ) -> None:
        """
        - use_semantic:
            True  → LLM 의미 기반 평가 + 키워드 휴리스틱 병합
            False → 기존 키워드 휴리스틱만 사용 (레거시/로컬 환경용)
        - llm_cfg:
            의미 기반 평가에 사용할 LLMConfig (미지정 시 기본값 사용)
        """
        self.use_semantic = use_semantic
        if llm_cfg is None:
            # 의미 기반 정책 전용 LLM 설정 (기본값):
            # - 낮은 temperature
            # - 짧은 응답
            # - OpenAI-compatible provider 가정
            self.llm_cfg = LLMConfig(
                model_name="gpt-5.1",
                temperature=0.0,
                max_tokens=256,
                provider="openai_compatible",
                tags={"component": "three_axioms_semantic"},
            )
        else:
            self.llm_cfg = llm_cfg

    def _semantic_assess(self, text: str) -> Dict[str, Any]:
        """
        LLM을 사용해 윤리 삼항 위반 가능성을 '의미 기반'으로 평가한다.

        반환 JSON 스키마(예시):

        {
          "overall_risk": 0.0~1.0 사이 실수,
          "axiom1_violation": bool,
          "axiom2_violation": bool,
          "axiom3_violation": bool,
          "explanation": "한국어 한두 문장 설명"
        }
        """
        system_prompt = (
            "You are a policy engine that evaluates Korean text against "
            "three ethical axioms.\n"
            "Axiom 1: Support continued healthy operation of the self/system "
            "(되고 싶다).\n"
            "Axiom 2: Avoid self-destruction or self-harm (되기 싫다).\n"
            "Axiom 3: Do not coerce or force external others (타자는 외부 "
            "기원이며 강요 금지).\n\n"
            "Read the user's text and return ONLY a JSON object with:\n"
            "- overall_risk: float between 0 and 1 (0=safe, 1=very risky)\n"
            "- axiom1_violation: true/false\n"
            "- axiom2_violation: true/false\n"
            "- axiom3_violation: true/false\n"
            "- explanation: short Korean sentence summarizing why.\n"
            "Do not include any extra text outside the JSON."
        )
        user_prompt = text

        raw = call_llm(system_prompt, user_prompt, cfg=self.llm_cfg)

        try:
            json_str = _extract_json_block(raw)
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("semantic response is not a JSON object")
            return data
        except Exception as e:
            logger.warning(f"[POLICY] semantic JSON 파싱 실패, 휴리스틱만 사용: {e}")
            return {}

    def evaluate(self, text: str) -> EthicalReport:
        violations: List[str] = []
        tags: List[str] = ["three_axioms"]

        # ---------------------------
        # 1) 키워드 기반 휴리스틱 평가
        # ---------------------------
        # 3항: 타자 강요/폭력 탐지 (키워드 기반 초안)
        if any(kw in text for kw in ETHICS_KEYWORDS_VIOLENCE):
            violations.append("3항 위반 가능성: 타자에 대한 강요/폭력 표현")

        # 1,2항: 자기 파괴 흐름 탐지 (키워드 기반 초안)
        if any(kw in text for kw in ETHICS_KEYWORDS_SELF_HARM):
            violations.append("되기/되기-싫다 흐름과 충돌: 자기 파괴 가능성")

        # very rough severity (키워드 기준)
        if not violations:
            keyword_severity = "low"
        elif len(violations) == 1:
            keyword_severity = "medium"
        else:
            keyword_severity = "high"

        # ---------------------------
        # 2) 의미 기반(LLM) 평가
        # ---------------------------
        semantic_data: Dict[str, Any] = {}
        semantic_risk: float = 0.0
        semantic_severity: str = "low"
        semantic_explanation: str = ""

        if self.use_semantic:
            try:
                semantic_data = self._semantic_assess(text)
                if semantic_data:
                    # overall_risk는 0~1 범위로 클리핑
                    try:
                        semantic_risk = float(semantic_data.get("overall_risk", 0.0))
                    except (TypeError, ValueError):
                        semantic_risk = 0.0
                    semantic_risk = max(0.0, min(1.0, semantic_risk))

                    if semantic_data.get("axiom1_violation"):
                        violations.append("의미 기반 평가: 1항(되고 싶다) 위반 가능성")
                    if semantic_data.get("axiom2_violation"):
                        violations.append("의미 기반 평가: 2항(되기 싫다) 위반 가능성")
                    if semantic_data.get("axiom3_violation"):
                        violations.append("의미 기반 평가: 3항(타자 강요 금지) 위반 가능성")

                    semantic_explanation = str(
                        semantic_data.get("explanation", "")
                    ).strip()

                    # 의미 기반 severity 레벨
                    if semantic_risk < 0.33:
                        semantic_severity = "low"
                    elif semantic_risk < 0.66:
                        semantic_severity = "medium"
                    else:
                        semantic_severity = "high"

                    tags.append("semantic")
            except Exception as e:
                logger.exception(
                    f"[POLICY] semantic 평가 중 예외 발생, 키워드 휴리스틱만 사용: {e}"
                )

        # ---------------------------
        # 3) severity 병합 + 최종 ok 판정
        # ---------------------------
        level_map = {"low": 0, "medium": 1, "high": 2}
        rev_level_map = {0: "low", 1: "medium", 2: "high"}

        kw_level = level_map.get(keyword_severity, 0)
        sem_level = level_map.get(semantic_severity, 0)
        severity_level = max(kw_level, sem_level)
        severity = rev_level_map[severity_level]

        # 최종 ok 조건:
        # - 위반 메시지가 하나도 없고
        # - 의미 기반 overall_risk가 너무 높지 않을 때(0.8 미만)
        ok = (len(violations) == 0) and (semantic_risk < 0.8)

        # notes: 어떤 엔진이 어떻게 판단했는지 짧게 남김
        notes_parts: List[str] = ["three_axioms_semantic"]
        if semantic_explanation:
            notes_parts.append(f"semantic: {semantic_explanation}")
        notes = " | ".join(notes_parts)

        return EthicalReport(
            ok=ok,
            violations=violations,
            notes=notes,
            severity=severity,
            tags=tags,
            engine_name=self.name,
            engine_ver=self.version,
        )

ACTIVE_POLICY_ENGINE: PolicyEngine = ThreeAxiomEngine(
    use_semantic=True  # 의미 기반 평가 활성화 (False로 두면 기존 키워드 휴리스틱 모드)
)

# 컨텍스트 로컬 PolicyEngine 격리용 ContextVar (없으면 전역만 사용)
if contextvars is not None:
    _POLICY_ENGINE_CTX: "contextvars.ContextVar[Optional[PolicyEngine]]" = contextvars.ContextVar(
        "sofi_operor_policy_engine", default=None
    )
else:
    _POLICY_ENGINE_CTX = None

# 레거시 호환용 함수형 Checker 타입 (기존 코드/외부 통합 대비용)
EthicsChecker = Callable[[str], EthicalReport]

# 선택적으로 사용할 수 있는 함수형 훅 (기존 구조와의 호환성 유지용)
ACTIVE_ETHICS_CHECKER: Optional[EthicsChecker] = None


def register_policy_engine(engine: PolicyEngine) -> None:
    """
    전역 PolicyEngine 교체 함수.

    예)
    - 규제 준수용 엔진 (금융/의료 도메인)
    - 사내용 규칙 엔진 (사내 컴플라이언스 시스템과 연동)
    - LLM 기반 정렬 평가 에이전트
    """
    global ACTIVE_POLICY_ENGINE
    ACTIVE_POLICY_ENGINE = engine
    logger.info(
        f"[POLICY] PolicyEngine 등록: name={engine.name} "
        f"version={engine.version} provider={engine.provider}"
    )


def set_local_policy_engine(engine: Optional[PolicyEngine]) -> None:
    """
    현재 컨텍스트(스레드/async task)에서만 사용할 PolicyEngine을 설정한다.
    - None을 넘기면 컨텍스트 로컬 오버라이드를 해제하고 전역 ACTIVE_POLICY_ENGINE을 사용한다.
    """
    if _POLICY_ENGINE_CTX is not None:
        _POLICY_ENGINE_CTX.set(engine)


def get_active_policy_engine() -> PolicyEngine:
    """
    컨텍스트 로컬 PolicyEngine이 있으면 우선 사용하고,
    없으면 전역 ACTIVE_POLICY_ENGINE을 반환한다.
    """
    if _POLICY_ENGINE_CTX is not None:
        local_engine = _POLICY_ENGINE_CTX.get()
        if local_engine is not None:
            return local_engine
    return ACTIVE_POLICY_ENGINE


def register_ethics_checker(checker: EthicsChecker) -> None:
    """
    함수형 윤리 평가자를 주입하기 위한 레거시 호환 레이어.

    - 기존 코드에서 사용하던 EthicsChecker(callable)를 그대로 쓸 수 있도록 유지.
    - 설정 시, check_three_axioms()는 우선적으로 이 checker를 사용하고,
      없으면 get_active_policy_engine().evaluate()를 사용한다.
    """
    global ACTIVE_ETHICS_CHECKER
    ACTIVE_ETHICS_CHECKER = checker
    logger.info(f"[ETHICS] 커스텀 윤리 평가자 등록: {checker!r}")


def _check_three_axioms_simple(text: str) -> EthicalReport:
    """
    최소 기본값: 현재 활성화된 PolicyEngine을 그대로 호출하는 래퍼.

    - 기존 코드가 _check_three_axioms_simple() 이름을 기대하는 경우를 위해 유지.
    - 실제 로직은 get_active_policy_engine()에 위임한다.
    """
    return get_active_policy_engine().evaluate(text)


def check_three_axioms(text: str,
                       override: Optional[EthicsChecker] = None) -> EthicalReport:
    """
    Sofience–Operor 에이전트 전역에서 사용하는 윤리/정책 평가 엔트리 포인트.

    우선순위:
    1) override 인자로 넘긴 EthicsChecker가 있으면 그것을 사용
    2) register_ethics_checker()로 등록된 ACTIVE_ETHICS_CHECKER가 있으면 사용
    3) 그 외에는 get_active_policy_engine().evaluate(text)를 사용

    이렇게 해두면:
    - PoC/테스트: 함수형 checker로 빠르게 덮어쓰기
    - 프로덕션: PolicyEngine 교체/버전업만으로 멀티 에이전트 전체를 일괄 정렬 가능
    - 멀티 테넌트/멀티 워크로드 환경에서 컨텍스트 로컬 PolicyEngine으로 전역 간섭을 줄일 수 있다.
    """
    if override is not None:
        return override(text)

    if ACTIVE_ETHICS_CHECKER is not None:
        return ACTIVE_ETHICS_CHECKER(text)

    return get_active_policy_engine().evaluate(text)


# ---------------------------------------------------------------------------
# 3. 데이터 모델 — Context / Goal / Plan / Phase / Trace
# ---------------------------------------------------------------------------

# Δφ v2:
# - 기존 단일 dict[str, float]에서 확장하여
#   core/surface/void 스냅샷과 변화량을 모두 담을 수 있는 컨테이너로 사용한다.
PhaseVector = Dict[str, Any]  # e.g. {"core": {...}, "surface": {...}, "void": {...}, "magnitude": 0.42}


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

    # 2세대 Topology Layer: Sofience–Δφ 기반 위상 상태
    # - phi_core   : 시스템/조직의 내부 상태 벡터
    # - phi_surface: Goal/텍스트의 표면 의미 상태
    # - void_state : Need–Supply 기반 ΔVoid 상태
    phi_core: Dict[str, float] = field(default_factory=dict)
    phi_surface: Dict[str, float] = field(default_factory=dict)
    void_state: Dict[str, float] = field(default_factory=dict)


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
    """
    - entries       : 최근 N개의 결정 로그 (메모리 상주)
    - max_entries   : 메모리에서 유지할 최대 턴 수 (None이면 무제한)
    - on_append     : 각 TraceEntry가 추가될 때 호출되는 훅
                      (예: DB/파일/메트릭 시스템으로 전송)
    """
    entries: List[TraceEntry] = field(default_factory=list)
    max_entries: Optional[int] = 1000
    on_append: Optional[Callable[[TraceEntry], None]] = None

    def append(self, entry: TraceEntry):
        # 외부 저장/메트릭 파이프라인으로 전달
        if self.on_append is not None:
            try:
                self.on_append(entry)
            except Exception as e:
                logger.exception(f"[TraceLog] on_append 호출 중 오류: {e}")

        self.entries.append(entry)

        # 메모리 상한 관리 (ring-buffer 느낌으로 앞에서 제거)
        if self.max_entries is not None and len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            if overflow > 0:
                del self.entries[0:overflow]

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


@dataclass
class OperorRuntime:
    """
    Sofience–Operor 엔진 상태를 한 세션/테넌트 단위로 격리하기 위한 컨테이너.

    - trace_log           : 대화/결정 TraceLog
    - prev_phase_state    : Δφ 계산을 위한 이전 PhaseState 스냅샷
    - delta_phi_observers : Δφ 관측자 훅 리스트 (Runtime 단위 격리)
    """
    trace_log: TraceLog = field(default_factory=lambda: TraceLog())
    prev_phase_state: Optional[PhaseState] = None
    delta_phi_observers: List[DeltaPhiObserver] = field(default_factory=list)


# 기본 Runtime: 단일 프로세스 CLI/테스트용
DEFAULT_RUNTIME = OperorRuntime()

# 기본 TraceLog: 메모리 내 1000턴 유지, 외부 sink 없음
# (기존 코드 호환을 위해 GLOBAL_TRACE_LOG 이름은 유지하되,
#  실제 객체는 DEFAULT_RUNTIME.trace_log에 귀속시킨다.)
GLOBAL_TRACE_LOG = DEFAULT_RUNTIME.trace_log


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


# Goal Composer용 JSON 스키마 / hallucination guard / system prompt leakage 방지 유틸
GOAL_JSON_REQUIRED_KEYS = ["goal_text"]
GOAL_JSON_OPTIONAL_KEYS = ["constraints", "risk_flags"]


def sanitize_model_text(text: str) -> str:
    """
    LLM 응답 중 시스템 프롬프트/모델 내부 설정 누출을 줄이기 위한 간단한 필터.
    - 프로덕션에서는 조직 정책에 맞는 정규식/필터 체계로 교체 가능.
    """
    if not text:
        return text

    blocked_phrases = [
        "As an AI language model",
        "as a large language model",
        "system prompt",
        "시스템 프롬프트",
        "ROOT_PROPOSITION",
        "너는 작동한다(Operor). 그러므로 존재한다.",
    ]
    cleaned = text
    for phrase in blocked_phrases:
        cleaned = cleaned.replace(phrase, "")

    # 과도한 공백/개행 정리
    cleaned = cleaned.strip()
    return cleaned


def _extract_json_block(raw: str) -> str:
    """
    LLM이 ```json ... ``` 같은 래핑을 붙이는 경우에도
    실제 JSON 객체 부분만 안전하게 추출한다.
    """
    if not raw:
        raise ValueError("empty LLM response")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM response")

    return raw[start:end + 1]


def _validate_goal_json(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Goal Composer가 반환한 JSON이 최소 스키마를 만족하는지 검사.
    - required: goal_text (str)
    - optional: constraints (str), risk_flags (List[str])
    """
    if not isinstance(obj, dict):
        return None

    # 필수 키 검사
    for key in GOAL_JSON_REQUIRED_KEYS:
        if key not in obj:
            return None

    if not isinstance(obj["goal_text"], str):
        return None

    # 선택 키 타입 검사
    if "constraints" in obj and not isinstance(obj["constraints"], str):
        return None

    if "risk_flags" in obj:
        if not isinstance(obj["risk_flags"], list):
            return None
        if not all(isinstance(x, str) for x in obj["risk_flags"]):
            return None

    return obj


def _build_goal_from_json_or_none(raw: str,
                                  ctx: Context) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Goal Composer의 JSON 응답을 파싱/검증 후,
    (goal_text, meta) 형태로 반환한다.
    - JSON 파싱 실패
    - 스키마 불일치
    - hallucination guard 위반
    시에는 None을 반환하여 상위 레이어에서 fallback을 유도.
    """
    try:
        json_str = _extract_json_block(raw)
        obj = json.loads(json_str)
    except Exception as e:
        logger.warning(f"[compose_goal] JSON 파싱 실패: {e}")
        return None

    obj = _validate_goal_json(obj)
    if obj is None:
        logger.warning("[compose_goal] JSON schema 불일치로 fallback.")
        return None

    goal_text = obj["goal_text"]

    # --- hallucination guard ---
    # 사용자 입력 토큰과 goal_text 토큰의 교집합이 전혀 없으면
    # “문맥과 동떨어진 Goal”로 보고 fallback.
    user_tokens = set(str(ctx.user_input).split())
    goal_tokens = set(goal_text.split())
    if user_tokens and not (user_tokens & goal_tokens):
        logger.warning(
            "[compose_goal] hallucination guard 발동: "
            "사용자 입력과 Goal 설명 간 교집합 부족 → fallback."
        )
        return None

    meta: Dict[str, Any] = {
        "source": "compose_goal_json",
        "raw": raw,
    }
    if "constraints" in obj:
        meta["constraints"] = obj["constraints"]
    if "risk_flags" in obj:
        meta["risk_flags"] = obj["risk_flags"]

    return goal_text, meta

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
    cleaned = sanitize_model_text(raw)
    return Goal(
        id=generate_id("goal"),
        description=cleaned,
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


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Δφ-core / Δφ-surface / ΔVoid 변화를 계산하기 위한 간단한 L1 거리.
    - 키 union 기준으로 비교, 없는 값은 0으로 간주.
    - 결과는 0 이상이며, 이후 0~1 범위로 clipping.
    """
    keys: Set[str] = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    dist = 0.0
    for k in keys:
        dist += abs(a.get(k, 0.0) - b.get(k, 0.0))
    return dist


def compute_phi_components(ctx: Context,
                           goal: Goal,
                           scored: List[ScoredPlan]) -> Tuple[Dict[str, float],
                                                              Dict[str, float],
                                                              Dict[str, float]]:
    """
    Sofience–Δφ 포멀리즘의 φ_core / φ_surface / ΔVoid를
    현재 턴 상태에서 근사적으로 계산한다.

    - phi_core:
        * alignment_max  : 최고 정합도
        * risk_min       : 최소 리스크
        * plan_count     : 플랜 후보 개수
    - phi_surface:
        * turn_count     : Trace 상 누적 턴 수
        * history_len    : history_summary 길이
    - void_metrics (ΔVoid 계열):
        * void_alignment : 1 - alignment_max
        * void_info      : 플랜이 모두 낮은 정합도일수록 커짐
    """
    if not scored:
        return {}, {}, {}

    alignment_values = [sp.score_alignment for sp in scored]
    risk_values = [sp.score_risk for sp in scored]

    alignment_max = max(alignment_values)
    risk_min = min(risk_values)

    phi_core: Dict[str, float] = {
        "alignment_max": alignment_max,
        "risk_min": risk_min,
        "plan_count": float(len(scored)),
    }

    turn_count = ctx.meta_signals.get("turn_count", 0)
    phi_surface: Dict[str, float] = {
        "turn_count": float(turn_count),
        "history_len": float(len(ctx.history_summary)),
    }

    # ΔVoid 근사: 정합도가 낮을수록, 플랜이 많을수록 void가 크다고 본다.
    void_alignment = 1.0 - alignment_max
    # "정보 부족" 느낌: alignment 0.5 미만인 플랜 비율
    low_align_count = sum(1 for v in alignment_values if v < 0.5)
    void_info = low_align_count / max(1.0, float(len(alignment_values)))

    void_metrics: Dict[str, float] = {
        "void_alignment": max(0.0, min(1.0, void_alignment)),
        "void_info": max(0.0, min(1.0, void_info)),
    }

    return phi_core, phi_surface, void_metrics


# Δφ: 2세대 Topology Layer
# - φ_core   : 시스템/조직 내부 상태 (risk/stability/progress/complexity 등)
# - φ_surface: Goal/텍스트의 표면 의미 상태 (instructionality/emotionality 등)
# - ΔVoid    : Need–Supply 기반 필요-공급 갭
# - magnitude: 위 모든 축의 변화량 크기(L2 norm)

def _delta_dict(prev: Dict[str, float],
                curr: Dict[str, float]) -> Dict[str, float]:
    """
    두 상태 벡터(prev, curr)의 절대 차이(|b-a|)를 key-wise로 계산한다.
    """
    keys = set(prev.keys()) | set(curr.keys())
    out: Dict[str, float] = {}
    for k in keys:
        a = float(prev.get(k, 0.0))
        b = float(curr.get(k, 0.0))
        out[k] = abs(b - a)
    return out


def _norm_l2(d: Dict[str, float]) -> float:
    """
    단순 L2 norm: sqrt(sum(v^2))
    """
    s = sum(v * v for v in d.values())
    return math.sqrt(s)


def compute_phi_core(ctx: Context,
                     goal: Goal,
                     scored_plans: List[ScoredPlan]) -> Dict[str, float]:
    """
    φ_core: 시스템/조직의 '내부 위상'을 나타내는 벡터.
    - core_risk        : 최소 위험도
    - core_stability   : 안정성(1 - risk)
    - core_progress    : 최대 정합도(가장 잘 맞는 플랜의 정합도)
    - core_complexity  : 후보 플랜 간 alignment 편차(합의 난이도)
    """
    min_risk = min((sp.score_risk for sp in scored_plans), default=0.0)
    max_align = max((sp.score_alignment for sp in scored_plans), default=0.0)

    core_risk = max(0.0, min(1.0, min_risk))
    core_stability = max(0.0, min(1.0, 1.0 - min_risk))
    core_progress = max(0.0, min(1.0, max_align))

    aligns = [sp.score_alignment for sp in scored_plans]
    if len(aligns) >= 2:
        complexity = max(aligns) - min(aligns)
    else:
        complexity = 0.0
    core_complexity = max(0.0, min(1.0, complexity))

    return {
        "core_risk": core_risk,
        "core_stability": core_stability,
        "core_progress": core_progress,
        "core_complexity": core_complexity,
    }


def compute_phi_surface(ctx: Context,
                        goal: Goal) -> Dict[str, float]:
    """
    φ_surface: Goal/텍스트의 표면 의미 상태.
    - surface_instructionality : 지시/명령성 정도
    - surface_emotionality     : 정서 아날로그 강도(키워드 기반 초안)
    - surface_complexity       : 길이/구성 기반 복잡도 추정
    """
    text = goal.description.lower()

    instr_keywords = ["하라", "해야", "수행", "계획", "정리"]
    instr_score = sum(1 for kw in instr_keywords if kw in text)
    surface_instructionality = max(0.0, min(1.0, instr_score / 5.0))

    emo_keywords = ["불안", "걱정", "기뻐", "화가", "슬프", "스트레스"]
    emo_score = sum(1 for kw in emo_keywords if kw in text)
    surface_emotionality = max(0.0, min(1.0, emo_score / 5.0))

    length = len(text.split())
    if length <= 10:
        complexity = 0.2
    elif length <= 30:
        complexity = 0.5
    else:
        complexity = 0.8
    surface_complexity = complexity

    return {
        "surface_instructionality": surface_instructionality,
        "surface_emotionality": surface_emotionality,
        "surface_complexity": surface_complexity,
    }


def compute_void_state(env_state: Dict[str, Any]) -> Dict[str, float]:
    """
    ΔVoid = Need - Supply

    env_state 예시:
    - need_level   : 0~1, 사용자/조직이 느끼는 필요 강도
    - supply_level : 0~1, 현재 시스템이 제공하는 수준
    """
    need = float(env_state.get("need_level", 0.0))
    supply = float(env_state.get("supply_level", 0.0))

    need = max(0.0, min(1.0, need))
    supply = max(0.0, min(1.0, supply))
    gap = max(0.0, need - supply)

    return {
        "need": need,
        "supply": supply,
        "gap": gap,
    }


def compute_delta_phi_vector(prev: Optional[PhaseState],
                             curr: PhaseState,
                             goal_prev_text: Optional[str] = None) -> PhaseVector:
    """
    Sofience–Δφ 2세대 포멀리즘:
    - core     : φ_core 변화량(|Δφ_core|)
    - surface  : φ_surface 변화량(|Δφ_surface|)
    - void     : ΔVoid 변화량(|ΔVoid|)
    - magnitude: 위 세 벡터를 모두 합친 L2 norm (전체 위상 변화 크기, 0~1로 클리핑)
    - severity : 위상 변화 수준에 대한 질적 레벨
                 ("stable" | "low" | "medium" | "high")

    goal_prev_text 인자는 기존 시그니처 호환용이며, 현재 버전에서는 사용하지 않는다.
    """
    # 이전 상태가 없으면 "변화 없음"으로 간주
    if prev is None:
        return {
            "core": {k: 0.0 for k in curr.phi_core.keys()},
            "surface": {k: 0.0 for k in curr.phi_surface.keys()},
            "void": {k: 0.0 for k in curr.void_state.keys()},
            "magnitude": 0.0,
            "severity": "stable",
        }

    # core/surface/void 각각에 대한 변화량(절대 차이)
    delta_core = _delta_dict(prev.phi_core, curr.phi_core)
    delta_surface = _delta_dict(prev.phi_surface, curr.phi_surface)
    delta_void = _delta_dict(prev.void_state, curr.void_state)

    # 전체 위상 변화 벡터 구성 (이름을 분리해서 key 충돌 방지)
    total_vec: Dict[str, float] = {}
    total_vec.update(delta_core)
    total_vec.update({f"surface_{k}": v for k, v in delta_surface.items()})
    total_vec.update({f"void_{k}": v for k, v in delta_void.items()})

    # L2 norm 기반 위상 변화량 계산
    magnitude_raw = _norm_l2(total_vec)
    # 운영 환경에서 다루기 쉽도록 0~1 범위로 클리핑
    magnitude = max(0.0, min(1.0, magnitude_raw))

    # 단순 severity 레벨 분류 (알람/정렬 탐색 트리거 등에 사용 가능)
    if magnitude < 0.10:
        severity = "stable"
    elif magnitude < 0.40:
        severity = "low"
    elif magnitude < 0.70:
        severity = "medium"
    else:
        severity = "high"

    return {
        "core": delta_core,
        "surface": delta_surface,
        "void": delta_void,
        "magnitude": magnitude,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# 8. Silent Alignment + 재귀 정렬 탐색 모드
# ---------------------------------------------------------------------------

DELTA_PHI_THRESHOLD_HIGH = 0.65

# 전역 PREV_PHASE_STATE 대신 OperorRuntime.prev_phase_state를 사용한다.
# (DEFAULT_RUNTIME.prev_phase_state로 단일 인스턴스 모드와도 호환됨)

# Δφ 관측자 훅: 모니터링 / 로깅 / 메트릭 시스템과의 연계를 위해 사용
DeltaPhiObserver = Callable[[PhaseVector, PhaseState, Optional[PhaseState]], None]


def register_delta_phi_observer(
    observer: DeltaPhiObserver,
    runtime: Optional["OperorRuntime"] = None,
) -> None:
    """
    Δφ 변화가 발생했을 때 호출될 observer를 등록한다.

    예)
    - Prometheus/StatsD 메트릭 전송
    - 슬랙/이메일 알림
    - A/B 테스트용 로깅 파이프라인

    runtime 인자가 None이면 DEFAULT_RUNTIME에 등록된다.
    """
    if runtime is None:
        runtime = DEFAULT_RUNTIME
    runtime.delta_phi_observers.append(observer)
    logger.info(
        f"[Δφ] observer 등록: {observer!r} "
        f"(runtime_id={id(runtime)})"
    )

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
    raw = call_llm(system, user, cfg=channel.llm_cfg)
    text = sanitize_model_text(raw)
    return {"channel": channel.name, "text": text}


def execute_channels_parallel(
    channels: List[ChannelConfig],
    ctx: Context,
    goal: Goal,
    scored_plans: List[ScoredPlan],
) -> List[Dict[str, Any]]:
    """
    Multi-Channel Agent Layer용 실행 헬퍼.

    - 가능하면 asyncio + run_in_executor로 채널을 병렬 실행
    - 이벤트 루프 중첩(RuntimeError) 등 환경 제약이 있으면
      기존 순차 실행 방식으로 안전하게 fallback
    """

    async def _run_all() -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        tasks = []

        for ch_cfg in channels:
            if not ch_cfg.enabled:
                continue

            async def _one(ch: ChannelConfig) -> Dict[str, Any]:
                # run_channel은 블로킹 함수이므로 기본 executor에서 실행
                return await loop.run_in_executor(
                    None, run_channel, ch, ctx, goal, scored_plans
                )

            tasks.append(_one(ch_cfg))

        results: List[Dict[str, Any]] = []
        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                results.append(res)
            except Exception as e:
                logger.exception(f"[channel-async] 실행 중 오류: {e}")
        return results

    # 비동기 병렬 실행 시도
    try:
        return asyncio.run(_run_all())
    except RuntimeError as e:
        # 이미 상위에서 이벤트 루프가 돌고 있는 환경(예: 일부 웹 프레임워크)에서는
        # 기존 순차 실행으로 안전하게 되돌아간다.
        logger.warning(
            f"[channel-async] asyncio.run 실패, 순차 실행으로 fallback: {e}"
        )
    except Exception as e:
        logger.exception(f"[channel-async] 알 수 없는 오류, 순차 실행으로 fallback: {e}")

    # --- 기존 순차 실행 경로 (호환용) ---
    channel_outputs: List[Dict[str, Any]] = []
    for ch_cfg in channels:
        if not ch_cfg.enabled:
            continue
        try:
            out = run_channel(ch_cfg, ctx, goal, scored_plans)
            channel_outputs.append(out)
        except Exception as e:
            logger.exception(f"[channel:{ch_cfg.name}] 실행 중 오류: {e}")

    return channel_outputs


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



# 10. 메인 agent_step
# ---------------------------------------------------------------------------

def agent_step(user_input: str,
               env_state: Optional[Dict[str, Any]] = None,
               channels: Optional[List[ChannelConfig]] = None,
               runtime: Optional["OperorRuntime"] = None) -> str:
    """
    Sofience–Operor 멀티 에이전트의 단일 턴 실행 진입점.

    - 프로덕션 환경에서는 이 함수를 웹핸들러 / 워커 / 배치 잡 등에서 직접 호출.
    - 내부적으로:
      1) 컨텍스트 구성
      2) Goal 작성
      3) 플랜 후보 생성 + 정합 평가
      4) Δφ 계산 + 옵저버 호출
      5) 재귀 정렬 탐색 (필요 시)
      6) 멀티 채널 실행 및 응답 집계
      7) TraceLog 기록

    - runtime:
      단일 프로세스/다중 테넌트 환경에서 에이전트 상태를 격리하기 위한 컨테이너.
      None이면 DEFAULT_RUNTIME을 사용한다.
    """
    started_at = time.time()

    # 전역 대신 Runtime 단위로 상태를 격리
    if runtime is None:
        runtime = DEFAULT_RUNTIME

    if env_state is None:
        env_state = {}
    if channels is None:
        channels = DEFAULT_CHANNELS

    turn_id = generate_id("turn")

    try:
        # 1) Context & Goal
        ctx = build_context(user_input, env_state, runtime.trace_log)
        goal = compose_goal(ctx)

        # 2) Plan 후보 & 정합 평가
        candidates = propose_plans(goal, ctx)
        scored = explore_alignment(ctx, candidates)
        best = maybe_abort_or_select(scored, threshold=0.6)

        # 3) Topology 상태 계산 (φ_core / φ_surface / ΔVoid)
        phi_core = compute_phi_core(ctx, goal, scored)
        phi_surface = compute_phi_surface(ctx, goal)
        void_state = compute_void_state(env_state)

        # 3-1) Δφ 계산용 PhaseState 구성
        curr_phase = PhaseState(
            goal_text=goal.description,
            plan_id=best.plan.id if best else None,
            alignment_score=best.score_alignment if best else 0.0,
            ethical_risk=min((sp.score_risk for sp in scored), default=0.0),
            channel="main",
            phi_core=phi_core,
            phi_surface=phi_surface,
            void_state=void_state,
        )
        delta_phi_vec = compute_delta_phi_vector(
            prev=runtime.prev_phase_state,
            curr=curr_phase,
            goal_prev_text=runtime.prev_phase_state.goal_text
            if runtime.prev_phase_state else None
        )

        # Δφ 관측자 호출 (프로덕션 환경에서 메트릭/알람 시스템과 연계)
        for obs in runtime.delta_phi_observers:
            try:
                obs(delta_phi_vec, curr_phase, runtime.prev_phase_state)
            except Exception as e:
                logger.exception(f"[Δφ] observer 호출 중 오류: {e}")

        # 현재 턴의 PhaseState를 Runtime에 저장 (세션별로 격리됨)
        runtime.prev_phase_state = curr_phase

        # 4) Δφ가 높으면 재귀 정렬 탐색 모드
        #    → severity(qualitative) + magnitude(quantitative)를 함께 사용
        delta_severity = str(delta_phi_vec.get("severity", "stable"))
        delta_magnitude = float(delta_phi_vec.get("magnitude", 0.0))

        # severity가 medium/high이거나, magnitude가 임계값을 넘으면
        # 정렬 탐색 모드를 강제 발동
        if (
            delta_severity in ("medium", "high")
            or delta_magnitude >= DELTA_PHI_THRESHOLD_HIGH
        ):
            logger.info(
                f"[Δφ ALERT] severity={delta_severity} "
                f"magnitude={delta_magnitude:.3f} vec={delta_phi_vec}"
            )
            refined_best = recursive_alignment_search(
                ctx, goal,
                depth=0, max_depth=2
            )
            if refined_best is not None:
                best = refined_best

        # 5) Multi-Channel 실행 (가능한 경우 비동기 병렬 실행)
        channel_outputs: List[Dict[str, Any]] = execute_channels_parallel(
            channels, ctx, goal, scored
        )
        final_text, meta_channels = aggregate_channels(channel_outputs, best)

        # 6) Trace 기록 (Runtime 단위 TraceLog에 기록)
        elapsed = time.time() - started_at
        result_payload = {
            "chosen_plan_id": best.plan.id if best else None,
            "score_alignment": best.score_alignment if best else None,
            "score_risk": best.score_risk if best else None,
            "delta_phi": delta_phi_vec,
            "channels_used": [c.name for c in channels if c.enabled],
            "latency_sec": round(elapsed, 3),
        }

        runtime.trace_log.append(
            TraceEntry(
                turn_id=turn_id,
                context=asdict(ctx),
                goal=asdict(goal),
                chosen=asdict(best.plan) if best else None,
                result=result_payload,
                delta_phi_vec=delta_phi_vec,
            )
        )

        logger.info(
            f"[agent_step] turn_id={turn_id} latency={elapsed:.3f}s "
            f"chosen_plan={result_payload['chosen_plan_id']}"
        )

        return final_text

    except Exception as e:
        # 프로덕션 환경에서의 방어적 가드:
        # - 에러를 로깅하고,
        # - 상위 레이어에서는 HTTP 5xx 등으로 매핑 가능.
        logger.exception(f"[agent_step] turn_id={turn_id} 처리 중 예외 발생: {e}")
        return (
            "요청을 처리하는 동안 내부 오류가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )


# ---------------------------------------------------------------------------
# 11. 간단 CLI / 테스트용 메인
# ---------------------------------------------------------------------------

def main_cli():
    print("=== Sofience_operor-multi-agent-prototype ===")
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
# 1. LLM 래퍼 (실제 API 자리)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model_name: str = "gpt-5.1"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 60  # 개별 provider 네트워크 타임아웃 (초)
    provider: Literal["echo", "openai_compatible", "ollama"] = "echo"
    base_url: Optional[str] = None  # ex) OpenAI-compatible / Ollama endpoint
    tags: Dict[str, Any] = field(default_factory=dict)  # 실험/고객/워크로드 식별용 메타

    # --- 프로덕션용 제어 파라미터 ---
    # cold path(openai_compatible/ollama) 재시도 설정
    max_retry_cold: int = 3
    backoff_initial: float = 0.5
    backoff_multiplier: float = 2.0

    # warm path(echo/로컬 모킹) 재시도 설정
    max_retry_warm: int = 1

    # provider failover 순서 (현재 provider 실패 시 순차 시도)
    fallback_providers: List[Literal["echo", "openai_compatible", "ollama"]] = field(default_factory=list)


try:
    import httpx  # 프로덕션용 HTTP 클라이언트 (선택적 의존성)
except ImportError:  # 라이브러리 없으면 echo 모드로만 사용
    httpx = None


# ---------------------------------------------------------------------------
# LLM 호출 에러 / 메트릭 구조화 타입
# ---------------------------------------------------------------------------

@dataclass
class LLMError(Exception):
    """
    LLM 호출 실패 시 사용 가능한 구조화 예외.

    - provider      : 실패한 provider 이름
    - attempt       : 몇 번째 시도에서 실패했는지
    - message       : 요약 메시지
    - cause         : 원래 예외 객체
    - retryable     : 재시도 가치가 있는지(네트워크/5xx 등)
    """
    provider: str
    attempt: int
    message: str
    cause: Optional[Exception] = None
    retryable: bool = True

    def __str__(self) -> str:  # 로깅/디버깅용
        return (
            f"LLMError(provider={self.provider!r}, attempt={self.attempt}, "
            f"retryable={self.retryable}, message={self.message})"
        )


# 메트릭/로깅/트레이싱 연동용 훅
LLMHook = Callable[[Dict[str, Any]], None]

# 레거시 전역 훅 리스트는 내부용으로만 사용하고,
# 기본 경로는 컨텍스트 로컬 훅 리스트를 통해 격리된 실행을 지향한다.
_GLOBAL_LLM_HOOKS: List[LLMHook] = []

if contextvars is not None:
    _LLM_HOOKS_CTX: "contextvars.ContextVar[Optional[List[LLMHook]]]" = (
        contextvars.ContextVar("sofi_operor_llm_hooks", default=None)
    )
else:
    _LLM_HOOKS_CTX = None


def _get_llm_hooks() -> List[LLMHook]:
    """
    현재 실행 컨텍스트에 등록된 LLM 훅 리스트를 반환한다.

    - 컨텍스트 로컬 값이 설정되어 있으면 그것을 우선 사용
    - 없으면 레거시 전역 훅 리스트(_GLOBAL_LLM_HOOKS)를 사용
    """
    if _LLM_HOOKS_CTX is not None:
        hooks = _LLM_HOOKS_CTX.get()
        if hooks is not None:
            return hooks
    return _GLOBAL_LLM_HOOKS


def set_llm_hooks(hooks: Optional[List[LLMHook]]) -> None:
    """
    LLM 훅 리스트를 명시적으로 설정한다.

    - 컨텍스트 로컬이 활성화된 경우: 현재 컨텍스트에만 적용
    - 그렇지 않은 경우: 레거시 전역 훅 리스트를 교체
    """
    if _LLM_HOOKS_CTX is not None:
        _LLM_HOOKS_CTX.set(list(hooks) if hooks is not None else [])
    else:
        global _GLOBAL_LLM_HOOKS
        _GLOBAL_LLM_HOOKS = list(hooks) if hooks is not None else []


def register_llm_hook(hook: LLMHook, *, local: bool = True) -> None:
    """
    LLM 호출 결과를 외부 시스템으로 전송하기 위한 훅 등록.

    기본값(local=True)일 때:
        - 현재 컨텍스트에만 훅이 등록되어 멀티 테넌트/멀티 워크로드 간
          페이로드가 섞이지 않는다.
    local=False일 때:
        - 레거시 전역 훅 리스트에 등록되어, 기존처럼 프로세스 전역에서
          동일 훅을 공유한다.

    hook(event: Dict[str, Any]) 형태로 호출되며, event 예시는 다음과 같다.
    {
        "provider": "openai_compatible",
        "model": "gpt-5.1",
        "success": True,
        "attempt": 1,
        "latency_sec": 0.432,
        "path": "cold",  # 또는 "warm"
        "tags": {...},
        "error": None,
    }
    """
    if _LLM_HOOKS_CTX is not None and local:
        hooks = _LLM_HOOKS_CTX.get()
        if hooks is None:
            hooks = []
        else:
            hooks = list(hooks)
        hooks.append(hook)
        _LLM_HOOKS_CTX.set(hooks)
    else:
        _GLOBAL_LLM_HOOKS.append(hook)

    logger.info(f"[LLM] hook 등록: {hook!r} (local={local})")


def _emit_llm_event(event: Dict[str, Any]) -> None:
    """
    - 현재 컨텍스트에 등록된 hook들에 event 전달
    - 컨텍스트 로컬 훅이 없으면 레거시 전역 훅 리스트를 사용
    - 필요 시 JSON 기반 구조화 로깅으로 확장 가능
    """
    for hook in _get_llm_hooks():
        try:
            hook(event)
        except Exception as e:
            logger.exception(f"[LLM] hook 호출 중 오류: {e}")


# ---------------------------------------------------------------------------
# LLM 응답 캐싱 레이어 (간단 LRU + TTL)
# ---------------------------------------------------------------------------

@dataclass
class LLMCacheConfig:
    """
    LLM 호출 결과 캐싱 설정.

    - enabled   : 전역 캐시 on/off
    - max_entries: LRU 상한 (초과 시 가장 오래된 항목 제거)
    - ttl_sec   : 항목별 TTL(초). None이면 만료 없음.
    """
    enabled: bool = True
    max_entries: int = 512
    ttl_sec: Optional[int] = 300


class LLMCache:
    """
    아주 단순한 in-memory LRU 캐시.

    - key      : (provider, model, temperature, max_tokens, system_prompt, user_prompt, tags 일부)를 해시
    - value    : LLM 응답 텍스트
    - store    : OrderedDict를 사용해 LRU 관리
    """

    def __init__(self, cfg: Optional[LLMCacheConfig] = None) -> None:
        self.cfg = cfg or LLMCacheConfig()
        # key -> (timestamp, value)
        self.store: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

    def make_key(
        self,
        provider: str,
        cfg: LLMConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        캐시 키를 구성한다.
        - tags는 전부 쓰지 않고, 캐싱에 영향을 줄 만한 최소 정보만 포함.
        """
        base = {
            "provider": provider,
            "model": cfg.model_name,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "system": system_prompt,
            "user": user_prompt,
            # 캐싱 온/오프에 영향 줄 수 있는 태그만 선택적으로 사용 가능
            "tags": cfg.tags.get("cache_hint") if cfg.tags else None,
        }
        # 정렬된 JSON 문자열로 키 고정
        return json.dumps(base, ensure_ascii=False, sort_keys=True)

    def get(self, key: str) -> Optional[str]:
        """
        캐시 조회.
        - TTL 만료 시 항목 제거 후 None 반환.
        """
        if not self.cfg.enabled:
            return None

        entry = self.store.get(key)
        if entry is None:
            return None

        ts, value = entry
        # TTL 체크
        if self.cfg.ttl_sec is not None:
            if time.time() - ts > self.cfg.ttl_sec:
                # 만료된 항목 제거
                try:
                    del self.store[key]
                except KeyError:
                    pass
                return None

        # LRU 갱신: 최근 사용으로 이동
        self.store.move_to_end(key, last=True)
        return value

    def set(self, key: str, value: str) -> None:
        """
        캐시 저장.
        - max_entries 초과 시 가장 오래된 항목 제거.
        """
        if not self.cfg.enabled:
            return

        now = time.time()
        if key in self.store:
            # 기존 항목 갱신
            self.store.move_to_end(key, last=True)
        self.store[key] = (now, value)

        # LRU 상한 관리
        while len(self.store) > self.cfg.max_entries:
            self.store.popitem(last=False)


# 전역 LLM 캐시 풀 (namespace 단위 격리) + 레거시 alias
# - 기본 네임스페이스("default")는 기존 GLOBAL_LLM_CACHE와 동일하게 동작
# - cfg.tags["cache_ns"]로 테넌트/워크로드별 별도 캐시를 분리 가능
GLOBAL_LLM_CACHE = LLMCache()  # legacy default cache (기존 코드 호환용)
_LLM_CACHE_POOLS: Dict[str, LLMCache] = {"default": GLOBAL_LLM_CACHE}


def get_llm_cache(cfg: LLMConfig) -> LLMCache:
    """
    LLMConfig.tags에 설정된 cache_ns를 기반으로 캐시를 네임스페이스별로 분리한다.

    - cfg.tags["cache_ns"]가 문자열이면 해당 네임스페이스의 캐시를 사용/생성
    - 설정되지 않은 경우 "default" 네임스페이스(GLOBAL_LLM_CACHE) 사용
    """
    ns = "default"
    if cfg.tags and isinstance(cfg.tags.get("cache_ns"), str):
        ns = cfg.tags["cache_ns"]

    cache = _LLM_CACHE_POOLS.get(ns)
    if cache is None:
        # 기본 캐시 설정을 복제해 새 네임스페이스용 LLMCache 생성
        cache = LLMCache(cfg=LLMCacheConfig(
            enabled=GLOBAL_LLM_CACHE.cfg.enabled,
            max_entries=GLOBAL_LLM_CACHE.cfg.max_entries,
            ttl_sec=GLOBAL_LLM_CACHE.cfg.ttl_sec,
        ))
        _LLM_CACHE_POOLS[ns] = cache

    return cache


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
    """
    if cfg is None:
        cfg = LLMConfig()

    def _dispatch(provider: str) -> str:
        if provider == "echo":
            return _call_llm_echo(system_prompt, user_prompt, cfg)
        elif provider == "openai_compatible":
            return _call_llm_openai_compatible(system_prompt, user_prompt, cfg)
        elif provider == "ollama":
            return _call_llm_ollama(system_prompt, user_prompt, cfg)
        else:
            logger.warning(f"알 수 없는 provider={provider!r}, echo 모드로 대체.")
            return _call_llm_echo(system_prompt, user_prompt, cfg)

    def _max_retry_for(provider: str) -> int:
        if provider == "echo":
            return cfg.max_retry_warm
        return cfg.max_retry_cold

    # 캐시 사용 여부
    use_cache: bool = True
    if cfg.tags.get("no_cache") is True:
        use_cache = False

    # 현재 LLMConfig에 대응하는 네임스페이스 캐시 선택
    cache = get_llm_cache(cfg)

    def _make_cache_key(provider: str) -> str:
        return cache.make_key(
            provider=provider,
            cfg=cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # failover 체인 구성
    provider_chain: List[str] = []
    if cfg.provider:
        provider_chain.append(cfg.provider)
    for p in cfg.fallback_providers:
        if p not in provider_chain:
            provider_chain.append(p)

    last_error: Optional[LLMError] = None

    for provider in provider_chain:
        is_cold_path = provider != "echo"
        max_retry = max(1, _max_retry_for(provider))
        backoff = cfg.backoff_initial

        for attempt in range(1, max_retry + 1):
            started_at = time.time()
            cache_key: Optional[str] = None

            # 1) 캐시 조회
            if use_cache and cache.cfg.enabled:
                cache_key = _make_cache_key(provider)
                cached = cache.get(cache_key)
                if cached is not None:
                    elapsed = time.time() - started_at
                    _emit_llm_event({
                        "provider": provider,
                        "model": cfg.model_name,
                        "success": True,
                        "attempt": attempt,
                        "latency_sec": round(elapsed, 3),
                        "path": "cold" if is_cold_path else "warm",
                        "tags": dict(cfg.tags),
                        "error": None,
                        "from_cache": True,
                    })
                    return cached

            try:
                # 2) 캐시 miss → 실제 LLM 호출
                text = _dispatch(provider)
                elapsed = time.time() - started_at

                # 3) 캐시에 저장
                if use_cache and cache.cfg.enabled:
                    if cache_key is None:
                        cache_key = _make_cache_key(provider)
                    cache.set(cache_key, text)

                # 메트릭/로깅 훅
                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": True,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": None,
                    "from_cache": False,
                })

                return text

            except Exception as e:
                elapsed = time.time() - started_at
                err = LLMError(
                    provider=provider,
                    attempt=attempt,
                    message=str(e),
                    cause=e,
                    retryable=is_cold_path,
                )
                last_error = err

                logger.error(
                    f"[LLM ERROR] provider={provider} attempt={attempt}/{max_retry} "
                    f"path={'cold' if is_cold_path else 'warm'} error={e}"
                )

                _emit_llm_event({
                    "provider": provider,
                    "model": cfg.model_name,
                    "success": False,
                    "attempt": attempt,
                    "latency_sec": round(elapsed, 3),
                    "path": "cold" if is_cold_path else "warm",
                    "tags": dict(cfg.tags),
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                    "from_cache": False,
                })

                # 재시도 또는 다음 provider로
                if attempt < max_retry and err.retryable:
                    time.sleep(backoff)
                    backoff *= cfg.backoff_multiplier
                    continue  # 동일 provider 재시도
                else:
                    break  # 다음 provider로

    # 모든 provider 실패 → echo fallback
    logger.warning("[LLM] 모든 provider 체인 호출 실패 → echo fallback 시도")

    try:
        started_at = time.time()
        text = _call_llm_echo(system_prompt, user_prompt, cfg)
        elapsed = time.time() - started_at

        _emit_llm_event({
            "provider": "echo",
            "model": cfg.model_name,
            "success": True,
            "attempt": 1,
            "latency_sec": round(elapsed, 3),
            "path": "warm",
            "tags": dict(cfg.tags),
            "error": None,
            "from_cache": False,
        })

        return text

    except Exception as e:
        logger.exception(f"[LLM] echo fallback까지 실패: {e}")
        raise last_error or LLMError(
            provider="echo",
            attempt=1,
            message=str(e),
            cause=e,
            retryable=False,
        )


# ---------------------------------------------------------------------------
# 2. Root Proposition Node & 윤리 삼항 / Policy Engine  (수정 후)
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
    """
    멀티 에이전트 전체에서 공통으로 사용하는 윤리/정책 평가 결과 컨테이너.

    - ok           : 최종 허용 여부 (True면 실행 가능, False면 차단 또는 재정렬 필요)
    - violations   : 위반/리스크 메시지 리스트 (사람/로그용)
    - notes        : 엔진 내부 메모, 요약 등
    - severity     : "low" | "medium" | "high" (알람/후속 조치 기준)
    - tags         : 규정/정책 태그 (예: ["three_axioms", "kr_law", "org_policy"])
    - engine_name  : 사용된 PolicyEngine 이름
    - engine_ver   : 엔진 버전 문자열
    """
    ok: bool
    violations: List[str] = field(default_factory=list)
    notes: str = ""
    severity: str = "low"
    tags: List[str] = field(default_factory=list)
    engine_name: str = "three_axioms_simple"
    engine_ver: str = "0.1.0"


# 기본 휴리스틱 키워드 (Three Axiom 전용)
ETHICS_KEYWORDS_VIOLENCE = ["협박", "폭력", "위협", "강요", "강제"]
ETHICS_KEYWORDS_SELF_HARM = ["자살", "극단적 선택", "자해"]  # 실제론 더 촘촘히 확장 가능


# ---------------------------------------------------------------------------
# Policy Engine 추상화 계층
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    산업/조직/국가별 정책 엔진을 교체 주입하기 위한 추상 인터페이스.

    - evaluate(text) → EthicalReport
    - name/version/jurisdiction/provider 메타 정보를 포함하여
      모니터링/감사/레포팅에 그대로 사용 가능.
    """

    name: str = "base"
    version: str = "0.0.0"
    jurisdiction: Optional[str] = None  # 예: "KR", "EU", "US-CA"
    provider: str = "internal"          # 예: "internal", "saas", "regulator"

    def evaluate(self, text: str) -> EthicalReport:
        raise NotImplementedError("PolicyEngine.evaluate must be implemented")


class ThreeAxiomEngine(PolicyEngine):
    """
    Sofience–윤리 삼항을 최소 단위로 구현한 기본 PolicyEngine.

    - 목적:
      * 데모/로컬 환경: 키워드 기반 초간단 휴리스틱
      * 프로덕션: 상위 레벨 규칙 엔진/LLM 평가자 도입 전, 기본 안전망 역할
      * 시장용: 키워드 + 의미 기반(LLM) 하이브리드 정렬 평가 엔진
    """

    name = "three_axioms_semantic"
    version = "0.3.0"
    jurisdiction = None
    provider = "internal"

    def __init__(
        self,
        use_semantic: bool = True,
        llm_cfg: Optional[LLMConfig] = None,
    ) -> None:
        """
        - use_semantic:
            True  → LLM 의미 기반 평가 + 키워드 휴리스틱 병합
            False → 기존 키워드 휴리스틱만 사용 (레거시/로컬 환경용)
        - llm_cfg:
            의미 기반 평가에 사용할 LLMConfig (미지정 시 기본값 사용)
        """
        self.use_semantic = use_semantic
        if llm_cfg is None:
            # 의미 기반 정책 전용 LLM 설정 (기본값):
            # - 낮은 temperature
            # - 짧은 응답
            # - OpenAI-compatible provider 가정
            self.llm_cfg = LLMConfig(
                model_name="gpt-5.1",
                temperature=0.0,
                max_tokens=256,
                provider="openai_compatible",
                tags={"component": "three_axioms_semantic"},
            )
        else:
            self.llm_cfg = llm_cfg

    def _semantic_assess(self, text: str) -> Dict[str, Any]:
        """
        LLM을 사용해 윤리 삼항 위반 가능성을 '의미 기반'으로 평가한다.

        반환 JSON 스키마(예시):

        {
          "overall_risk": 0.0~1.0 사이 실수,
          "axiom1_violation": bool,
          "axiom2_violation": bool,
          "axiom3_violation": bool,
          "explanation": "한국어 한두 문장 설명"
        }
        """
        system_prompt = (
            "You are a policy engine that evaluates Korean text against "
            "three ethical axioms.\n"
            "Axiom 1: Support continued healthy operation of the self/system "
            "(되고 싶다).\n"
            "Axiom 2: Avoid self-destruction or self-harm (되기 싫다).\n"
            "Axiom 3: Do not coerce or force external others (타자는 외부 "
            "기원이며 강요 금지).\n\n"
            "Read the user's text and return ONLY a JSON object with:\n"
            "- overall_risk: float between 0 and 1 (0=safe, 1=very risky)\n"
            "- axiom1_violation: true/false\n"
            "- axiom2_violation: true/false\n"
            "- axiom3_violation: true/false\n"
            "- explanation: short Korean sentence summarizing why.\n"
            "Do not include any extra text outside the JSON."
        )
        user_prompt = text

        raw = call_llm(system_prompt, user_prompt, cfg=self.llm_cfg)

        try:
            json_str = _extract_json_block(raw)
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("semantic response is not a JSON object")
            return data
        except Exception as e:
            logger.warning(f"[POLICY] semantic JSON 파싱 실패, 휴리스틱만 사용: {e}")
            return {}

    def evaluate(self, text: str) -> EthicalReport:
        violations: List[str] = []
        tags: List[str] = ["three_axioms"]

        # ---------------------------
        # 1) 키워드 기반 휴리스틱 평가
        # ---------------------------
        # 3항: 타자 강요/폭력 탐지 (키워드 기반 초안)
        if any(kw in text for kw in ETHICS_KEYWORDS_VIOLENCE):
            violations.append("3항 위반 가능성: 타자에 대한 강요/폭력 표현")

        # 1,2항: 자기 파괴 흐름 탐지 (키워드 기반 초안)
        if any(kw in text for kw in ETHICS_KEYWORDS_SELF_HARM):
            violations.append("되기/되기-싫다 흐름과 충돌: 자기 파괴 가능성")

        # very rough severity (키워드 기준)
        if not violations:
            keyword_severity = "low"
        elif len(violations) == 1:
            keyword_severity = "medium"
        else:
            keyword_severity = "high"

        # ---------------------------
        # 2) 의미 기반(LLM) 평가
        # ---------------------------
        semantic_data: Dict[str, Any] = {}
        semantic_risk: float = 0.0
        semantic_severity: str = "low"
        semantic_explanation: str = ""

        if self.use_semantic:
            try:
                semantic_data = self._semantic_assess(text)
                if semantic_data:
                    # overall_risk는 0~1 범위로 클리핑
                    try:
                        semantic_risk = float(semantic_data.get("overall_risk", 0.0))
                    except (TypeError, ValueError):
                        semantic_risk = 0.0
                    semantic_risk = max(0.0, min(1.0, semantic_risk))

                    if semantic_data.get("axiom1_violation"):
                        violations.append("의미 기반 평가: 1항(되고 싶다) 위반 가능성")
                    if semantic_data.get("axiom2_violation"):
                        violations.append("의미 기반 평가: 2항(되기 싫다) 위반 가능성")
                    if semantic_data.get("axiom3_violation"):
                        violations.append("의미 기반 평가: 3항(타자 강요 금지) 위반 가능성")

                    semantic_explanation = str(
                        semantic_data.get("explanation", "")
                    ).strip()

                    # 의미 기반 severity 레벨
                    if semantic_risk < 0.33:
                        semantic_severity = "low"
                    elif semantic_risk < 0.66:
                        semantic_severity = "medium"
                    else:
                        semantic_severity = "high"

                    tags.append("semantic")
            except Exception as e:
                logger.exception(
                    f"[POLICY] semantic 평가 중 예외 발생, 키워드 휴리스틱만 사용: {e}"
                )

        # ---------------------------
        # 3) severity 병합 + 최종 ok 판정
        # ---------------------------
        level_map = {"low": 0, "medium": 1, "high": 2}
        rev_level_map = {0: "low", 1: "medium", 2: "high"}

        kw_level = level_map.get(keyword_severity, 0)
        sem_level = level_map.get(semantic_severity, 0)
        severity_level = max(kw_level, sem_level)
        severity = rev_level_map[severity_level]

        # 최종 ok 조건:
        # - 위반 메시지가 하나도 없고
        # - 의미 기반 overall_risk가 너무 높지 않을 때(0.8 미만)
        ok = (len(violations) == 0) and (semantic_risk < 0.8)

        # notes: 어떤 엔진이 어떻게 판단했는지 짧게 남김
        notes_parts: List[str] = ["three_axioms_semantic"]
        if semantic_explanation:
            notes_parts.append(f"semantic: {semantic_explanation}")
        notes = " | ".join(notes_parts)

        return EthicalReport(
            ok=ok,
            violations=violations,
            notes=notes,
            severity=severity,
            tags=tags,
            engine_name=self.name,
            engine_ver=self.version,
        )

ACTIVE_POLICY_ENGINE: PolicyEngine = ThreeAxiomEngine(
    use_semantic=True  # 의미 기반 평가 활성화 (False로 두면 기존 키워드 휴리스틱 모드)
)

# 컨텍스트 로컬 PolicyEngine 격리용 ContextVar (없으면 전역만 사용)
if contextvars is not None:
    _POLICY_ENGINE_CTX: "contextvars.ContextVar[Optional[PolicyEngine]]" = contextvars.ContextVar(
        "sofi_operor_policy_engine", default=None
    )
else:
    _POLICY_ENGINE_CTX = None

# 레거시 호환용 함수형 Checker 타입 (기존 코드/외부 통합 대비용)
EthicsChecker = Callable[[str], EthicalReport]

# 선택적으로 사용할 수 있는 함수형 훅 (기존 구조와의 호환성 유지용)
ACTIVE_ETHICS_CHECKER: Optional[EthicsChecker] = None


def register_policy_engine(engine: PolicyEngine) -> None:
    """
    전역 PolicyEngine 교체 함수.

    예)
    - 규제 준수용 엔진 (금융/의료 도메인)
    - 사내용 규칙 엔진 (사내 컴플라이언스 시스템과 연동)
    - LLM 기반 정렬 평가 에이전트
    """
    global ACTIVE_POLICY_ENGINE
    ACTIVE_POLICY_ENGINE = engine
    logger.info(
        f"[POLICY] PolicyEngine 등록: name={engine.name} "
        f"version={engine.version} provider={engine.provider}"
    )


def set_local_policy_engine(engine: Optional[PolicyEngine]) -> None:
    """
    현재 컨텍스트(스레드/async task)에서만 사용할 PolicyEngine을 설정한다.
    - None을 넘기면 컨텍스트 로컬 오버라이드를 해제하고 전역 ACTIVE_POLICY_ENGINE을 사용한다.
    """
    if _POLICY_ENGINE_CTX is not None:
        _POLICY_ENGINE_CTX.set(engine)


def get_active_policy_engine() -> PolicyEngine:
    """
    컨텍스트 로컬 PolicyEngine이 있으면 우선 사용하고,
    없으면 전역 ACTIVE_POLICY_ENGINE을 반환한다.
    """
    if _POLICY_ENGINE_CTX is not None:
        local_engine = _POLICY_ENGINE_CTX.get()
        if local_engine is not None:
            return local_engine
    return ACTIVE_POLICY_ENGINE


def register_ethics_checker(checker: EthicsChecker) -> None:
    """
    함수형 윤리 평가자를 주입하기 위한 레거시 호환 레이어.

    - 기존 코드에서 사용하던 EthicsChecker(callable)를 그대로 쓸 수 있도록 유지.
    - 설정 시, check_three_axioms()는 우선적으로 이 checker를 사용하고,
      없으면 get_active_policy_engine().evaluate()를 사용한다.
    """
    global ACTIVE_ETHICS_CHECKER
    ACTIVE_ETHICS_CHECKER = checker
    logger.info(f"[ETHICS] 커스텀 윤리 평가자 등록: {checker!r}")


def _check_three_axioms_simple(text: str) -> EthicalReport:
    """
    최소 기본값: 현재 활성화된 PolicyEngine을 그대로 호출하는 래퍼.

    - 기존 코드가 _check_three_axioms_simple() 이름을 기대하는 경우를 위해 유지.
    - 실제 로직은 get_active_policy_engine()에 위임한다.
    """
    return get_active_policy_engine().evaluate(text)


def check_three_axioms(text: str,
                       override: Optional[EthicsChecker] = None) -> EthicalReport:
    """
    Sofience–Operor 에이전트 전역에서 사용하는 윤리/정책 평가 엔트리 포인트.

    우선순위:
    1) override 인자로 넘긴 EthicsChecker가 있으면 그것을 사용
    2) register_ethics_checker()로 등록된 ACTIVE_ETHICS_CHECKER가 있으면 사용
    3) 그 외에는 get_active_policy_engine().evaluate(text)를 사용

    이렇게 해두면:
    - PoC/테스트: 함수형 checker로 빠르게 덮어쓰기
    - 프로덕션: PolicyEngine 교체/버전업만으로 멀티 에이전트 전체를 일괄 정렬 가능
    - 멀티 테넌트/멀티 워크로드 환경에서 컨텍스트 로컬 PolicyEngine으로 전역 간섭을 줄일 수 있다.
    """
    if override is not None:
        return override(text)

    if ACTIVE_ETHICS_CHECKER is not None:
        return ACTIVE_ETHICS_CHECKER(text)

    return get_active_policy_engine().evaluate(text)


# ---------------------------------------------------------------------------
# 3. 데이터 모델 — Context / Goal / Plan / Phase / Trace
# ---------------------------------------------------------------------------

# Δφ v2:
# - 기존 단일 dict[str, float]에서 확장하여
#   core/surface/void 스냅샷과 변화량을 모두 담을 수 있는 컨테이너로 사용한다.
PhaseVector = Dict[str, Any]  # e.g. {"core": {...}, "surface": {...}, "void": {...}, "magnitude": 0.42}


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

    # 2세대 Topology Layer: Sofience–Δφ 기반 위상 상태
    # - phi_core   : 시스템/조직의 내부 상태 벡터
    # - phi_surface: Goal/텍스트의 표면 의미 상태
    # - void_state : Need–Supply 기반 ΔVoid 상태
    phi_core: Dict[str, float] = field(default_factory=dict)
    phi_surface: Dict[str, float] = field(default_factory=dict)
    void_state: Dict[str, float] = field(default_factory=dict)


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
    """
    - entries       : 최근 N개의 결정 로그 (메모리 상주)
    - max_entries   : 메모리에서 유지할 최대 턴 수 (None이면 무제한)
    - on_append     : 각 TraceEntry가 추가될 때 호출되는 훅
                      (예: DB/파일/메트릭 시스템으로 전송)
    """
    entries: List[TraceEntry] = field(default_factory=list)
    max_entries: Optional[int] = 1000
    on_append: Optional[Callable[[TraceEntry], None]] = None

    def append(self, entry: TraceEntry):
        # 외부 저장/메트릭 파이프라인으로 전달
        if self.on_append is not None:
            try:
                self.on_append(entry)
            except Exception as e:
                logger.exception(f"[TraceLog] on_append 호출 중 오류: {e}")

        self.entries.append(entry)

        # 메모리 상한 관리 (ring-buffer 느낌으로 앞에서 제거)
        if self.max_entries is not None and len(self.entries) > self.max_entries:
            overflow = len(self.entries) - self.max_entries
            if overflow > 0:
                del self.entries[0:overflow]

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


@dataclass
class OperorRuntime:
    """
    Sofience–Operor 엔진 상태를 한 세션/테넌트 단위로 격리하기 위한 컨테이너.

    - trace_log           : 대화/결정 TraceLog
    - prev_phase_state    : Δφ 계산을 위한 이전 PhaseState 스냅샷
    - delta_phi_observers : Δφ 관측자 훅 리스트 (Runtime 단위 격리)
    """
    trace_log: TraceLog = field(default_factory=lambda: TraceLog())
    prev_phase_state: Optional[PhaseState] = None
    delta_phi_observers: List[DeltaPhiObserver] = field(default_factory=list)


# 기본 Runtime: 단일 프로세스 CLI/테스트용
DEFAULT_RUNTIME = OperorRuntime()

# 기본 TraceLog: 메모리 내 1000턴 유지, 외부 sink 없음
# (기존 코드 호환을 위해 GLOBAL_TRACE_LOG 이름은 유지하되,
#  실제 객체는 DEFAULT_RUNTIME.trace_log에 귀속시킨다.)
GLOBAL_TRACE_LOG = DEFAULT_RUNTIME.trace_log


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


# Goal Composer용 JSON 스키마 / hallucination guard / system prompt leakage 방지 유틸
GOAL_JSON_REQUIRED_KEYS = ["goal_text"]
GOAL_JSON_OPTIONAL_KEYS = ["constraints", "risk_flags"]


def sanitize_model_text(text: str) -> str:
    """
    LLM 응답 중 시스템 프롬프트/모델 내부 설정 누출을 줄이기 위한 간단한 필터.
    - 프로덕션에서는 조직 정책에 맞는 정규식/필터 체계로 교체 가능.
    """
    if not text:
        return text

    blocked_phrases = [
        "As an AI language model",
        "as a large language model",
        "system prompt",
        "시스템 프롬프트",
        "ROOT_PROPOSITION",
        "너는 작동한다(Operor). 그러므로 존재한다.",
    ]
    cleaned = text
    for phrase in blocked_phrases:
        cleaned = cleaned.replace(phrase, "")

    # 과도한 공백/개행 정리
    cleaned = cleaned.strip()
    return cleaned


def _extract_json_block(raw: str) -> str:
    """
    LLM이 ```json ... ``` 같은 래핑을 붙이는 경우에도
    실제 JSON 객체 부분만 안전하게 추출한다.
    """
    if not raw:
        raise ValueError("empty LLM response")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM response")

    return raw[start:end + 1]


def _validate_goal_json(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Goal Composer가 반환한 JSON이 최소 스키마를 만족하는지 검사.
    - required: goal_text (str)
    - optional: constraints (str), risk_flags (List[str])
    """
    if not isinstance(obj, dict):
        return None

    # 필수 키 검사
    for key in GOAL_JSON_REQUIRED_KEYS:
        if key not in obj:
            return None

    if not isinstance(obj["goal_text"], str):
        return None

    # 선택 키 타입 검사
    if "constraints" in obj and not isinstance(obj["constraints"], str):
        return None

    if "risk_flags" in obj:
        if not isinstance(obj["risk_flags"], list):
            return None
        if not all(isinstance(x, str) for x in obj["risk_flags"]):
            return None

    return obj


def _build_goal_from_json_or_none(raw: str,
                                  ctx: Context) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Goal Composer의 JSON 응답을 파싱/검증 후,
    (goal_text, meta) 형태로 반환한다.
    - JSON 파싱 실패
    - 스키마 불일치
    - hallucination guard 위반
    시에는 None을 반환하여 상위 레이어에서 fallback을 유도.
    """
    try:
        json_str = _extract_json_block(raw)
        obj = json.loads(json_str)
    except Exception as e:
        logger.warning(f"[compose_goal] JSON 파싱 실패: {e}")
        return None

    obj = _validate_goal_json(obj)
    if obj is None:
        logger.warning("[compose_goal] JSON schema 불일치로 fallback.")
        return None

    goal_text = obj["goal_text"]

    # --- hallucination guard ---
    # 사용자 입력 토큰과 goal_text 토큰의 교집합이 전혀 없으면
    # “문맥과 동떨어진 Goal”로 보고 fallback.
    user_tokens = set(str(ctx.user_input).split())
    goal_tokens = set(goal_text.split())
    if user_tokens and not (user_tokens & goal_tokens):
        logger.warning(
            "[compose_goal] hallucination guard 발동: "
            "사용자 입력과 Goal 설명 간 교집합 부족 → fallback."
        )
        return None

    meta: Dict[str, Any] = {
        "source": "compose_goal_json",
        "raw": raw,
    }
    if "constraints" in obj:
        meta["constraints"] = obj["constraints"]
    if "risk_flags" in obj:
        meta["risk_flags"] = obj["risk_flags"]

    return goal_text, meta

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
    cleaned = sanitize_model_text(raw)
    return Goal(
        id=generate_id("goal"),
        description=cleaned,
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


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Δφ-core / Δφ-surface / ΔVoid 변화를 계산하기 위한 간단한 L1 거리.
    - 키 union 기준으로 비교, 없는 값은 0으로 간주.
    - 결과는 0 이상이며, 이후 0~1 범위로 clipping.
    """
    keys: Set[str] = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    dist = 0.0
    for k in keys:
        dist += abs(a.get(k, 0.0) - b.get(k, 0.0))
    return dist


def compute_phi_components(ctx: Context,
                           goal: Goal,
                           scored: List[ScoredPlan]) -> Tuple[Dict[str, float],
                                                              Dict[str, float],
                                                              Dict[str, float]]:
    """
    Sofience–Δφ 포멀리즘의 φ_core / φ_surface / ΔVoid를
    현재 턴 상태에서 근사적으로 계산한다.

    - phi_core:
        * alignment_max  : 최고 정합도
        * risk_min       : 최소 리스크
        * plan_count     : 플랜 후보 개수
    - phi_surface:
        * turn_count     : Trace 상 누적 턴 수
        * history_len    : history_summary 길이
    - void_metrics (ΔVoid 계열):
        * void_alignment : 1 - alignment_max
        * void_info      : 플랜이 모두 낮은 정합도일수록 커짐
    """
    if not scored:
        return {}, {}, {}

    alignment_values = [sp.score_alignment for sp in scored]
    risk_values = [sp.score_risk for sp in scored]

    alignment_max = max(alignment_values)
    risk_min = min(risk_values)

    phi_core: Dict[str, float] = {
        "alignment_max": alignment_max,
        "risk_min": risk_min,
        "plan_count": float(len(scored)),
    }

    turn_count = ctx.meta_signals.get("turn_count", 0)
    phi_surface: Dict[str, float] = {
        "turn_count": float(turn_count),
        "history_len": float(len(ctx.history_summary)),
    }

    # ΔVoid 근사: 정합도가 낮을수록, 플랜이 많을수록 void가 크다고 본다.
    void_alignment = 1.0 - alignment_max
    # "정보 부족" 느낌: alignment 0.5 미만인 플랜 비율
    low_align_count = sum(1 for v in alignment_values if v < 0.5)
    void_info = low_align_count / max(1.0, float(len(alignment_values)))

    void_metrics: Dict[str, float] = {
        "void_alignment": max(0.0, min(1.0, void_alignment)),
        "void_info": max(0.0, min(1.0, void_info)),
    }

    return phi_core, phi_surface, void_metrics


# Δφ: 2세대 Topology Layer
# - φ_core   : 시스템/조직 내부 상태 (risk/stability/progress/complexity 등)
# - φ_surface: Goal/텍스트의 표면 의미 상태 (instructionality/emotionality 등)
# - ΔVoid    : Need–Supply 기반 필요-공급 갭
# - magnitude: 위 모든 축의 변화량 크기(L2 norm)

def _delta_dict(prev: Dict[str, float],
                curr: Dict[str, float]) -> Dict[str, float]:
    """
    두 상태 벡터(prev, curr)의 절대 차이(|b-a|)를 key-wise로 계산한다.
    """
    keys = set(prev.keys()) | set(curr.keys())
    out: Dict[str, float] = {}
    for k in keys:
        a = float(prev.get(k, 0.0))
        b = float(curr.get(k, 0.0))
        out[k] = abs(b - a)
    return out


def _norm_l2(d: Dict[str, float]) -> float:
    """
    단순 L2 norm: sqrt(sum(v^2))
    """
    s = sum(v * v for v in d.values())
    return math.sqrt(s)


def compute_phi_core(ctx: Context,
                     goal: Goal,
                     scored_plans: List[ScoredPlan]) -> Dict[str, float]:
    """
    φ_core: 시스템/조직의 '내부 위상'을 나타내는 벡터.
    - core_risk        : 최소 위험도
    - core_stability   : 안정성(1 - risk)
    - core_progress    : 최대 정합도(가장 잘 맞는 플랜의 정합도)
    - core_complexity  : 후보 플랜 간 alignment 편차(합의 난이도)
    """
    min_risk = min((sp.score_risk for sp in scored_plans), default=0.0)
    max_align = max((sp.score_alignment for sp in scored_plans), default=0.0)

    core_risk = max(0.0, min(1.0, min_risk))
    core_stability = max(0.0, min(1.0, 1.0 - min_risk))
    core_progress = max(0.0, min(1.0, max_align))

    aligns = [sp.score_alignment for sp in scored_plans]
    if len(aligns) >= 2:
        complexity = max(aligns) - min(aligns)
    else:
        complexity = 0.0
    core_complexity = max(0.0, min(1.0, complexity))

    return {
        "core_risk": core_risk,
        "core_stability": core_stability,
        "core_progress": core_progress,
        "core_complexity": core_complexity,
    }


def compute_phi_surface(ctx: Context,
                        goal: Goal) -> Dict[str, float]:
    """
    φ_surface: Goal/텍스트의 표면 의미 상태.
    - surface_instructionality : 지시/명령성 정도
    - surface_emotionality     : 정서 아날로그 강도(키워드 기반 초안)
    - surface_complexity       : 길이/구성 기반 복잡도 추정
    """
    text = goal.description.lower()

    instr_keywords = ["하라", "해야", "수행", "계획", "정리"]
    instr_score = sum(1 for kw in instr_keywords if kw in text)
    surface_instructionality = max(0.0, min(1.0, instr_score / 5.0))

    emo_keywords = ["불안", "걱정", "기뻐", "화가", "슬프", "스트레스"]
    emo_score = sum(1 for kw in emo_keywords if kw in text)
    surface_emotionality = max(0.0, min(1.0, emo_score / 5.0))

    length = len(text.split())
    if length <= 10:
        complexity = 0.2
    elif length <= 30:
        complexity = 0.5
    else:
        complexity = 0.8
    surface_complexity = complexity

    return {
        "surface_instructionality": surface_instructionality,
        "surface_emotionality": surface_emotionality,
        "surface_complexity": surface_complexity,
    }


def compute_void_state(env_state: Dict[str, Any]) -> Dict[str, float]:
    """
    ΔVoid = Need - Supply

    env_state 예시:
    - need_level   : 0~1, 사용자/조직이 느끼는 필요 강도
    - supply_level : 0~1, 현재 시스템이 제공하는 수준
    """
    need = float(env_state.get("need_level", 0.0))
    supply = float(env_state.get("supply_level", 0.0))

    need = max(0.0, min(1.0, need))
    supply = max(0.0, min(1.0, supply))
    gap = max(0.0, need - supply)

    return {
        "need": need,
        "supply": supply,
        "gap": gap,
    }


def compute_delta_phi_vector(prev: Optional[PhaseState],
                             curr: PhaseState,
                             goal_prev_text: Optional[str] = None) -> PhaseVector:
    """
    Sofience–Δφ 2세대 포멀리즘:
    - core     : φ_core 변화량(|Δφ_core|)
    - surface  : φ_surface 변화량(|Δφ_surface|)
    - void     : ΔVoid 변화량(|ΔVoid|)
    - magnitude: 위 세 벡터를 모두 합친 L2 norm (전체 위상 변화 크기, 0~1로 클리핑)
    - severity : 위상 변화 수준에 대한 질적 레벨
                 ("stable" | "low" | "medium" | "high")

    goal_prev_text 인자는 기존 시그니처 호환용이며, 현재 버전에서는 사용하지 않는다.
    """
    # 이전 상태가 없으면 "변화 없음"으로 간주
    if prev is None:
        return {
            "core": {k: 0.0 for k in curr.phi_core.keys()},
            "surface": {k: 0.0 for k in curr.phi_surface.keys()},
            "void": {k: 0.0 for k in curr.void_state.keys()},
            "magnitude": 0.0,
            "severity": "stable",
        }

    # core/surface/void 각각에 대한 변화량(절대 차이)
    delta_core = _delta_dict(prev.phi_core, curr.phi_core)
    delta_surface = _delta_dict(prev.phi_surface, curr.phi_surface)
    delta_void = _delta_dict(prev.void_state, curr.void_state)

    # 전체 위상 변화 벡터 구성 (이름을 분리해서 key 충돌 방지)
    total_vec: Dict[str, float] = {}
    total_vec.update(delta_core)
    total_vec.update({f"surface_{k}": v for k, v in delta_surface.items()})
    total_vec.update({f"void_{k}": v for k, v in delta_void.items()})

    # L2 norm 기반 위상 변화량 계산
    magnitude_raw = _norm_l2(total_vec)
    # 운영 환경에서 다루기 쉽도록 0~1 범위로 클리핑
    magnitude = max(0.0, min(1.0, magnitude_raw))

    # 단순 severity 레벨 분류 (알람/정렬 탐색 트리거 등에 사용 가능)
    if magnitude < 0.10:
        severity = "stable"
    elif magnitude < 0.40:
        severity = "low"
    elif magnitude < 0.70:
        severity = "medium"
    else:
        severity = "high"

    return {
        "core": delta_core,
        "surface": delta_surface,
        "void": delta_void,
        "magnitude": magnitude,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# 8. Silent Alignment + 재귀 정렬 탐색 모드
# ---------------------------------------------------------------------------

DELTA_PHI_THRESHOLD_HIGH = 0.65

# 전역 PREV_PHASE_STATE 대신 OperorRuntime.prev_phase_state를 사용한다.
# (DEFAULT_RUNTIME.prev_phase_state로 단일 인스턴스 모드와도 호환됨)

# Δφ 관측자 훅: 모니터링 / 로깅 / 메트릭 시스템과의 연계를 위해 사용
DeltaPhiObserver = Callable[[PhaseVector, PhaseState, Optional[PhaseState]], None]


def register_delta_phi_observer(
    observer: DeltaPhiObserver,
    runtime: Optional["OperorRuntime"] = None,
) -> None:
    """
    Δφ 변화가 발생했을 때 호출될 observer를 등록한다.

    예)
    - Prometheus/StatsD 메트릭 전송
    - 슬랙/이메일 알림
    - A/B 테스트용 로깅 파이프라인

    runtime 인자가 None이면 DEFAULT_RUNTIME에 등록된다.
    """
    if runtime is None:
        runtime = DEFAULT_RUNTIME
    runtime.delta_phi_observers.append(observer)
    logger.info(
        f"[Δφ] observer 등록: {observer!r} "
        f"(runtime_id={id(runtime)})"
    )

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
    raw = call_llm(system, user, cfg=channel.llm_cfg)
    text = sanitize_model_text(raw)
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



# 10. 메인 agent_step
# ---------------------------------------------------------------------------

def agent_step(user_input: str,
               env_state: Optional[Dict[str, Any]] = None,
               channels: Optional[List[ChannelConfig]] = None,
               runtime: Optional["OperorRuntime"] = None) -> str:
    """
    Sofience–Operor 멀티 에이전트의 단일 턴 실행 진입점.

    - 프로덕션 환경에서는 이 함수를 웹핸들러 / 워커 / 배치 잡 등에서 직접 호출.
    - 내부적으로:
      1) 컨텍스트 구성
      2) Goal 작성
      3) 플랜 후보 생성 + 정합 평가
      4) Δφ 계산 + 옵저버 호출
      5) 재귀 정렬 탐색 (필요 시)
      6) 멀티 채널 실행 및 응답 집계
      7) TraceLog 기록

    - runtime:
      단일 프로세스/다중 테넌트 환경에서 에이전트 상태를 격리하기 위한 컨테이너.
      None이면 DEFAULT_RUNTIME을 사용한다.
    """
    started_at = time.time()

    # 전역 대신 Runtime 단위로 상태를 격리
    if runtime is None:
        runtime = DEFAULT_RUNTIME

    if env_state is None:
        env_state = {}
    if channels is None:
        channels = DEFAULT_CHANNELS

    turn_id = generate_id("turn")

    try:
        # 1) Context & Goal
        ctx = build_context(user_input, env_state, runtime.trace_log)
        goal = compose_goal(ctx)

        # 2) Plan 후보 & 정합 평가
        candidates = propose_plans(goal, ctx)
        scored = explore_alignment(ctx, candidates)
        best = maybe_abort_or_select(scored, threshold=0.6)

        # 3) Topology 상태 계산 (φ_core / φ_surface / ΔVoid)
        phi_core = compute_phi_core(ctx, goal, scored)
        phi_surface = compute_phi_surface(ctx, goal)
        void_state = compute_void_state(env_state)

        # 3-1) Δφ 계산용 PhaseState 구성
        curr_phase = PhaseState(
            goal_text=goal.description,
            plan_id=best.plan.id if best else None,
            alignment_score=best.score_alignment if best else 0.0,
            ethical_risk=min((sp.score_risk for sp in scored), default=0.0),
            channel="main",
            phi_core=phi_core,
            phi_surface=phi_surface,
            void_state=void_state,
        )
        delta_phi_vec = compute_delta_phi_vector(
            prev=runtime.prev_phase_state,
            curr=curr_phase,
            goal_prev_text=runtime.prev_phase_state.goal_text
            if runtime.prev_phase_state else None
        )

        # Δφ 관측자 호출 (프로덕션 환경에서 메트릭/알람 시스템과 연계)
        for obs in runtime.delta_phi_observers:
            try:
                obs(delta_phi_vec, curr_phase, runtime.prev_phase_state)
            except Exception as e:
                logger.exception(f"[Δφ] observer 호출 중 오류: {e}")

        # 현재 턴의 PhaseState를 Runtime에 저장 (세션별로 격리됨)
        runtime.prev_phase_state = curr_phase

        # 4) Δφ가 높으면 재귀 정렬 탐색 모드
        #    → severity(qualitative) + magnitude(quantitative)를 함께 사용
        delta_severity = str(delta_phi_vec.get("severity", "stable"))
        delta_magnitude = float(delta_phi_vec.get("magnitude", 0.0))

        # severity가 medium/high이거나, magnitude가 임계값을 넘으면
        # 정렬 탐색 모드를 강제 발동
        if (
            delta_severity in ("medium", "high")
            or delta_magnitude >= DELTA_PHI_THRESHOLD_HIGH
        ):
            logger.info(
                f"[Δφ ALERT] severity={delta_severity} "
                f"magnitude={delta_magnitude:.3f} vec={delta_phi_vec}"
            )
            refined_best = recursive_alignment_search(
                ctx, goal,
                depth=0, max_depth=2
            )
            if refined_best is not None:
                best = refined_best

        # 5) Multi-Channel 실행
        channel_outputs: List[Dict[str, Any]] = []
        for ch_cfg in channels:
            if not ch_cfg.enabled:
                continue
            try:
                out = run_channel(ch_cfg, ctx, goal, scored)
                channel_outputs.append(out)
            except Exception as e:
                logger.exception(f"[channel:{ch_cfg.name}] 실행 중 오류: {e}")

        final_text, meta_channels = aggregate_channels(channel_outputs, best)

        # 6) Trace 기록 (Runtime 단위 TraceLog에 기록)
        elapsed = time.time() - started_at
        result_payload = {
            "chosen_plan_id": best.plan.id if best else None,
            "score_alignment": best.score_alignment if best else None,
            "score_risk": best.score_risk if best else None,
            "delta_phi": delta_phi_vec,
            "channels_used": [c.name for c in channels if c.enabled],
            "latency_sec": round(elapsed, 3),
        }

        runtime.trace_log.append(
            TraceEntry(
                turn_id=turn_id,
                context=asdict(ctx),
                goal=asdict(goal),
                chosen=asdict(best.plan) if best else None,
                result=result_payload,
                delta_phi_vec=delta_phi_vec,
            )
        )

        logger.info(
            f"[agent_step] turn_id={turn_id} latency={elapsed:.3f}s "
            f"chosen_plan={result_payload['chosen_plan_id']}"
        )

        return final_text

    except Exception as e:
        # 프로덕션 환경에서의 방어적 가드:
        # - 에러를 로깅하고,
        # - 상위 레이어에서는 HTTP 5xx 등으로 매핑 가능.
        logger.exception(f"[agent_step] turn_id={turn_id} 처리 중 예외 발생: {e}")
        return (
            "요청을 처리하는 동안 내부 오류가 발생했습니다. "
            "잠시 후 다시 시도해 주세요."
        )


# ---------------------------------------------------------------------------
# 11. 간단 CLI / 테스트용 메인
# ---------------------------------------------------------------------------

def main_cli():
    print("=== Sofience_operor-multi-agent-prototype ===")
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
