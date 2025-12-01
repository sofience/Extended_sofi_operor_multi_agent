**What should a market-ready, platform-agnostic multi-agent look like—one that avoids the vendor lock-in and self-collision issues of current systems?**

Let's figure this out together.

A multi-agent framework built on a philosophical foundation: *Operor ergo sum* (I operate, therefore I am).

---

## Why This Exists

Current multi-agent systems share common pain points:

| Problem | Typical Approach | Sofience-Operor |
|---------|------------------|-----------------|
| **Vendor lock-in** | Tied to OpenAI/Anthropic SDKs | Provider-agnostic with failover chains |
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
│  Context Engineering  →  Goal Composer  →  Plan Proposal│
│         │                     │                  │      │
│         ▼                     ▼                  ▼      │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Alignment Scoring + Ethics Check        │    │
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
│              │  severity ≥ medium?   │                  │
│              └───────────┬───────────┘                  │
│                     yes  │                              │
│                          ▼                              │
│           Recursive Alignment Search (max 2)            │
├─────────────────────────────────────────────────────────┤
│                    TraceLog + Hooks                     │
└─────────────────────────────────────────────────────────┘
│
▼
Final Response
---

## Key Features

### 🔄 Provider-Agnostic LLM Layer
```python
LLMConfig(
    provider="openai_compatible",  # or "ollama", "echo"
    fallback_providers=["ollama", "echo"],
    max_retry_cold=3,
    backoff_multiplier=2.0,
)
📐 Δφ (Phase Change) Detection
Measures state drift across three axes:
φ_core: Internal system state (risk, stability, progress)
φ_surface: Semantic surface (instructionality, emotionality)
ΔVoid: Need-supply gap
When magnitude ≥ 0.65 or severity ∈ {medium, high}:
→ Triggers recursive goal refinement
🛡️ Hybrid Policy Engine
ThreeAxiomEngine(
    use_semantic=True,  # LLM-based evaluation
    llm_cfg=LLMConfig(temperature=0.0),
)
Combines keyword heuristics with LLM semantic analysis for context-aware ethics checking.
📊 Full Observability
register_llm_hook(my_metrics_collector)
register_delta_phi_observer(my_alerting_system)
register_policy_engine(MyComplianceEngine())
Quick Start
Installation
git clone https://github.com/your-org/sofience-operor.git
cd sofience-operor
pip install httpx  # optional, for real LLM calls
Run in Echo Mode (no API needed)
python sofience_operor.py
Connect to Real LLM
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com"
Then modify the config:
ACTIVE_POLICY_ENGINE = ThreeAxiomEngine(
    use_semantic=True,
    llm_cfg=LLMConfig(
        provider="openai_compatible",
        model_name="gpt-4o",
    ),
)
Configuration
LLMConfig Options
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
PolicyEngine Options
Parameter
Default
Description
use_semantic
True
Enable LLM-based evaluation
llm_cfg
(internal)
LLM config for semantic eval
Extending
Custom Policy Engine
class MyComplianceEngine(PolicyEngine):
    name = "my_compliance"
    version = "1.0.0"
    jurisdiction = "KR"
    
    def evaluate(self, text: str) -> EthicalReport:
        # Your logic here
        return EthicalReport(ok=True, ...)

register_policy_engine(MyComplianceEngine())
Custom Δφ Observer
def my_alerter(delta_phi, curr_phase, prev_phase):
    if delta_phi["severity"] == "high":
        send_slack_alert(f"High drift detected: {delta_phi}")

register_delta_phi_observer(my_alerter)

Roadmap
[x] Core multi-channel agent
[x] Δφ topology layer
[x] Hybrid policy engine (keyword + semantic)
[ ] Session isolation (remove global state)
[ ] Async channel execution
[ ] Tool use layer
[ ] Long-term memory / RAG integration
[ ] pytest coverage

Philosophy Note

This project explores a question:

What if an AI agent's alignment wasn't imposed from outside, but emerged from its own operational logic?

The Three Axioms aren't restrictions—they're the foundation of coherent existence. An agent that wants to keep operating naturally avoids self-destruction and respects others' autonomy.

Whether this leads anywhere useful remains to be seen. That's why we're building in the open.

License
MIT

Contributing
Issues and PRs welcome. If you're interested in the philosophical foundations, check out the /docs folder (coming soon).

---
