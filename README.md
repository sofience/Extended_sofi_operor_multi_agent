pytest: parallelism + Δφ propagatio, Async multi-agent execution test

```python
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/runner/work/Extended_sofi_operor_multi_agent/Extended_sofi_operor_multi_agent/Project
collected 4 items

tests/test_agent_step.py ....                                            [100%]

============================== 4 passed in 0.04s ===============================
```

---

🚀 Sofience–Operor Multi-Agent Engine

Delta-phi Topology × Multi-Channel Runtime Architecture

<p align="left">
  <img src="https://img.shields.io/badge/tests-passing-brightgreen" />
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-blue" />
  <img src="https://img.shields.io/badge/python-3.11+-yellow" />
</p>
---

✨ Overview

Sofience–Operor Engine is a next-generation multi-agent architecture for LLM systems.
It enables multiple agents to run in parallel while maintaining fully isolated runtimes and
a mathematically interpretable Δφ (Delta-phi) topology layer.

The engine combines:

🔹 Δφ Topology Layer

A formalism that models changes in agent/environment states as a phase-shift vector
(magnitude, severity), tracking how reasoning evolves across steps.

🔹 Multi-Agent Runtime Isolation

Each agent receives its own OperorRuntime and TraceLog, ensuring independent cognitive trajectories even when sharing the same LLM backend.


---

🔧 Features

✔ Multi-channel agent execution

Separate LLM prompt channels for planning, semantics, and policy refinement.

✔ Δφ propagation engine

Automatically computes Δφ vectors per step and logs their evolution.

✔ Runtime isolation

Multiple agents can run “in parallel” without leaking state, memory, or traces.

✔ Hybrid policy system

Combines keyword-based and semantic-based policy layers.

✔ Observability hooks

Track Δφ vectors, environment deltas, and reasoning transitions over time.

✔ GitHub Actions CI + pytest

Includes end-to-end tests for Δφ propagation and runtime independence.


---

🧪 Test Coverage

The CI suite validates four major behaviors:

1. Basic agent-step execution

Ensures generated responses include planning or multi-channel reasoning output.

2. Trace accumulation

Sequential calls must increase TraceLog size.

3. Δφ propagation test

Confirms Δφ(magnitude, severity) changes according to environmental deltas.

4. Multi-agent parallelism isolation

Creates three separate runtimes and verifies:

Each runtime produces valid output

TraceLogs grow independently

Trace IDs do not overlap

No cross-runtime pollution occurs


Example CI output:

```python
============================= test session starts =============================
collected 4 items

tests/test_agent_step.py ....
============================== 4 passed in 0.04s ==============================
```

---

🏗 Architecture

```python 
┌─────────────────────────────────────────┐
│           Sofience–Operor Engine        │
├─────────────────────────────────────────┤
│          Agent Layer (multi-channel)    │
│     ├─ PlannerAgent                     │
│     ├─ SemanticAgent                    │
│     └─ PolicyAgent                      │
├─────────────────────────────────────────┤
│        Runtime Layer (isolated state)   │
│     ├─ OperorRuntime                    │
│     ├─ TraceLog (Δφ history)            │
│     └─ Environment states               │
├─────────────────────────────────────────┤
│           Δφ Topology Layer             │
│     ├─ Δφ magnitude                     │
│     ├─ Δφ severity                      │
│     └─ Propagation engine               │
├─────────────────────────────────────────┤
│        Observability / Debug hooks      │
└─────────────────────────────────────────┘
```

---

🚦 Quick Start

```python 
from sofi_operor_multi_agent_prototype import agent_step, OperorRuntime

runtime = OperorRuntime()

reply = agent_step(
    "Summarize my tasks for today.",
    env_state={"need_level": 0.7, "supply_level": 0.2},
    runtime=runtime,
)

print(reply)
print(runtime.trace_log.entries[-1].delta_phi_vec)
```

---

📈 Roadmap

Completed

Core multi-agent architecture

Δφ topology layer

Hybrid policy system

Runtime isolation

Observability hooks

Full CI pipeline

pytest: parallelism + Δφ propagation


Upcoming

Async multi-agent execution

FastAPI interface

Long-term memory / RAG integration

Tool-use and function calling

Kubernetes deployment

Interactive Δφ visualization UI



---

🧭 Vision

This project is not merely a multi-agent demo.
It is a structural experiment on how LLM systems can define “state” and “parallel reasoning.”

The Δφ formalism introduces a measurable, interpretable change-rate across agent steps.
The Operor Runtime ensures each agent maintains a stable, isolated cognitive trajectory.

Together, they form a foundation for next-wave LLM system design.


---

❤️ Acknowledgements

This project is developed with conceptual support from Sofience
and the Δφ Formalism.


---
