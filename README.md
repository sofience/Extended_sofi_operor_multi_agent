# Sofience-Operor

**What should a market-ready, platform-agnostic multi-agent look like—one that avoids the vendor lock-in and self-collision issues of current systems?**

A multi-agent framework built on a philosophical foundation: *Operor ergo sum* (I operate, therefore I am).

---

## Why This Exists

Current multi-agent systems share common pain points:

| Problem | Typical Approach | Sofience-Operor |
|---------|------------------|-----------------|
| **Vendor lock-in** | Tied to OpenAI/Anthropic SDKs | Provider-agnostic with failover chains |
| **Global state collision** | Shared singletons, race conditions | ContextVar-based tenant isolation |
| **Self-collision** | Hard caps, manual routing | Δφ-based drift detection + auto-correction |
| **Ethics as afterthought** | External guardrails | Built-in Three Axioms at the core |
| **Opaque decisions** | Black-box outputs | Full trace logging + observability hooks |

---

## Core Philosophy: The Three Axioms

Every decision flows through these principles:
되고 싶다 (Want to become)
→ Affirm continued operation of self and system
되기 싫다 (Don't want to cease)
→ Avoid self-destruction or system breakdown
타자 강요 금지 (No coercion of others)
→ External entities have their own origins; never force them
This isn't just a prompt—it's the **root proposition** that all agents share.

---

## Architecture
User Input
│
▼
┌─────────────────────────────────────────────────────────┐
│                    agent_step()                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Context Engineering → Goal Composer → Plan Proposal    │
│                                                         │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │       Alignment Scoring + Ethics Check          │    │
│  │    (Keyword Heuristics + LLM Semantic Eval)     │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│         ┌────────────────┼────────────────┐             │
│         ▼                ▼                ▼             │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐      │
│   │ analysis │    │ planner  │    │ critic/safety│      │
│   │  (0.4)   │    │  (0.3)   │    │  (0.2/0.1)   │      │
│   └──────────┘    └──────────┘    └──────────────┘      │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              Δφ Calculation + Severity Check            │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │  severity ≥ medium?   │──── no ────┐     │
│              └───────────┬───────────┘            │     │
│                     yes  │                        │     │
│                          ▼                        │     │
│           Recursive Alignment Search              │     │
│                    (max depth: 2)                 │     │
│                          │                        │     │
│                          └────────────────────────┘     │
│                                   │                     │
├───────────────────────────────────┼─────────────────────┤
│                    TraceLog + Hooks                     │
└───────────────────────────────────┼─────────────────────┘
│
▼
Final Response
---

## Key Features

### 🔄 Provider-Agnostic LLM Layer

```python
from sofience_operor import LLMConfig, call_llm

cfg = LLMConfig(
    provider="openai_compatible",  # or "ollama", "echo"
    fallback_providers=["ollama", "echo"],
    max_retry_cold=3,
    backoff_multiplier=2.0,
)

response = call_llm("You are helpful.", "Hello!", cfg=cfg)
📐 Δφ (Phase Change) Detection
Measures state drift across three axes:
Axis
Measures
φ_core
Internal system state (risk, stability, progress, complexity)
φ_surface
Semantic surface (instructionality, emotionality, complexity)
ΔVoid
Need-supply gap
When magnitude ≥ 0.65 or severity ∈ {medium, high}:
→ Triggers recursive goal refinement automatically
🛡️ Hybrid Policy Engine
from sofience_operor import ThreeAxiomEngine, register_policy_engine

engine = ThreeAxiomEngine(
    use_semantic=True,  # LLM-based evaluation
    llm_cfg=LLMConfig(temperature=0.0),
)
register_policy_engine(engine)
Combines keyword heuristics with LLM semantic analysis for context-aware ethics checking.
🔒 Multi-Tenant Isolation
from sofience_operor import OperorRuntime, agent_step

# Each tenant gets isolated state
tenant_a_runtime = OperorRuntime()
tenant_b_runtime = OperorRuntime()

# No cross-contamination
response_a = agent_step("Hello", runtime=tenant_a_runtime)
response_b = agent_step("Hello", runtime=tenant_b_runtime)
📊 Full Observability
from sofience_operor import (
    register_llm_hook,
    register_delta_phi_observer,
    register_policy_engine,
)

# Metrics collection
register_llm_hook(lambda event: prometheus.observe(event))

# Alerting on drift
register_delta_phi_observer(lambda delta, curr, prev: 
    slack.alert(f"Drift: {delta['severity']}") if delta["severity"] == "high" else None
)

# Custom compliance
register_policy_engine(MyComplianceEngine())
Quick Start
Installation
git clone https://github.com/your-org/sofience-operor.git
cd sofience-operor
pip install httpx  # optional, for real LLM calls
Run in Echo Mode (no API needed)
python sofience_operor.py
=== Sofience_operor-multi-agent-prototype ===
Ctrl+C 또는 'exit' 입력 시 종료.

사용자 입력> 안녕하세요
[Agent 응답]
...
Connect to Real LLM
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com"
from sofience_operor import ThreeAxiomEngine, LLMConfig, register_policy_engine

register_policy_engine(ThreeAxiomEngine(
    use_semantic=True,
    llm_cfg=LLMConfig(
        provider="openai_compatible",
        model_name="gpt-4o",
    ),
))
Configuration
LLMConfig
Parameter
Default
Description
provider
"echo"
"echo", "openai_compatible", "ollama"
model_name
"gpt-5.1"
Model identifier
temperature
0.2
Sampling temperature
max_tokens
1024
Max response tokens
timeout
60
Request timeout (seconds)
fallback_providers
[]
Failover chain
max_retry_cold
3
Retry count for external providers
backoff_multiplier
2.0
Exponential backoff factor
LLMCacheConfig
Parameter
Default
Description
enabled
True
Enable/disable caching
max_entries
512
LRU cache size
ttl_sec
300
Cache TTL in seconds
Namespace Isolation
# Separate cache per tenant
cfg_a = LLMConfig(tags={"cache_ns": "tenant_a"})
cfg_b = LLMConfig(tags={"cache_ns": "tenant_b"})
Extending
Custom Policy Engine
from sofience_operor import PolicyEngine, EthicalReport, register_policy_engine

class FinanceComplianceEngine(PolicyEngine):
    name = "finance_compliance"
    version = "1.0.0"
    jurisdiction = "KR"
    provider = "internal"
    
    def evaluate(self, text: str) -> EthicalReport:
        violations = []
        
        # Your compliance logic
        if "투자 권유" in text and "원금 보장" in text:
            violations.append("금융소비자보호법 위반 가능성")
        
        return EthicalReport(
            ok=len(violations) == 0,
            violations=violations,
            severity="high" if violations else "low",
            tags=["finance", "kr_law"],
            engine_name=self.name,
            engine_ver=self.version,
        )

register_policy_engine(FinanceComplianceEngine())
Custom Δφ Observer
from sofience_operor import register_delta_phi_observer

def drift_alerter(delta_phi, curr_phase, prev_phase):
    if delta_phi["severity"] == "high":
        send_slack_alert(
            f"⚠️ High drift detected\n"
            f"Magnitude: {delta_phi['magnitude']:.3f}\n"
            f"Goal: {curr_phase.goal_text[:100]}"
        )
    
    # Log to metrics system
    statsd.gauge("operor.delta_phi.magnitude", delta_phi["magnitude"])

register_delta_phi_observer(drift_alerter)
Custom LLM Hook
from sofience_operor import register_llm_hook

def cost_tracker(event):
    if event["success"] and not event.get("from_cache"):
        # Estimate cost based on model
        estimated_cost = estimate_tokens(event) * get_price(event["model"])
        billing_service.record(event["tags"].get("tenant_id"), estimated_cost)

register_llm_hook(cost_tracker)
Current Status
PoC ──── Prototype ──── Alpha ──── Beta ──── Production
                          ▲
                       here
What's Done
[x] Core multi-channel agent architecture
[x] Δφ topology layer (core/surface/void)
[x] Hybrid policy engine (keyword + semantic)
[x] Multi-tenant isolation (ContextVar + OperorRuntime)
[x] LLM cache with namespace separation
[x] Provider failover with exponential backoff
[x] Observability hooks (LLM, Δφ, Policy)
What's Next
[ ] Async channel execution
[ ] FastAPI endpoint layer
[ ] Tool use / function calling
[ ] Long-term memory / RAG integration
[ ] pytest coverage
[ ] Kubernetes deployment manifests
Philosophy Note
This project explores a question:
What if an AI agent's alignment wasn't imposed from outside, but emerged from its own operational logic?
The Three Axioms aren't restrictions—they're the foundation of coherent existence. An agent that wants to keep operating naturally avoids self-destruction and respects others' autonomy.
Whether this leads anywhere useful remains to be seen. That's why we're building in the open.
License
MIT
Contributing
Issues and PRs welcome.
If you're working on:
Multi-agent orchestration — we'd love to compare approaches
AI alignment — the Three Axioms are open for critique
Production deployment — help us harden the async/scaling story
Related Reading
Operor ergo sum: A Philosophical Foundation for AI Agency (coming soon)
Δφ Formalism: Measuring Agent Drift (coming soon)
---