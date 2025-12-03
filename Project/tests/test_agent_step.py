import os
import sys

# tests 폴더 기준 한 단계 위(Project 디렉터리)를 import 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sofi_operor_multi_agent_prototype import agent_step, OperorRuntime


def test_agent_step_basic_runs():
    runtime = OperorRuntime()
    reply = agent_step(
        "간단하게 오늘 할 일을 정리해줘.",
        env_state={"need_level": 0.7, "supply_level": 0.2},
        runtime=runtime,
    )

    # 최소한 뭔가 비어 있지 않고, 채널/플랜 관련 텍스트가 들어오는지만 확인
    assert isinstance(reply, str)
    assert len(reply) > 0
    assert "채널" in reply or "플랜" in reply or "Sofience–Operor" in reply


def test_agent_step_trace_grows():
    runtime = OperorRuntime()

    # 두 번 연속 호출해서 TraceLog가 쌓이는지 확인
    agent_step("첫 번째 입력", runtime=runtime)
    agent_step("두 번째 입력", runtime=runtime)

    assert len(runtime.trace_log.entries) >= 2
    # Δφ 벡터가 최소한 구조적으로 존재하는지 확인
    last = runtime.trace_log.entries[-1]
    assert "magnitude" in (last.delta_phi_vec or {})
    assert "severity" in (last.delta_phi_vec or {})


# -------------------------------
# Δφ Propagation Test 추가됨
# -------------------------------
def test_delta_phi_propagation_env_change():
    runtime = OperorRuntime()

    # 1) baseline 환경
    agent_step(
        "baseline",
        env_state={"need_level": 0.5, "supply_level": 0.5},
        runtime=runtime,
    )

    # 2) 거의 같은 환경 → Δφ 변화 최소 예상
    agent_step(
        "same env again",
        env_state={"need_level": 0.5, "supply_level": 0.5},
        runtime=runtime,
    )

    # 3) 급격히 다른 환경 → Δφ가 달라져야 한다
    agent_step(
        "env changed",
        env_state={"need_level": 0.9, "supply_level": 0.1},
        runtime=runtime,
    )

    assert len(runtime.trace_log.entries) >= 3

    base = runtime.trace_log.entries[-3].delta_phi_vec or {}
    same = runtime.trace_log.entries[-2].delta_phi_vec or {}
    changed = runtime.trace_log.entries[-1].delta_phi_vec or {}

    # Δφ 벡터 기본 키 체크
    for vec in (base, same, changed):
        assert "magnitude" in vec
        assert "severity" in vec

    # 환경 변화가 있었으므로 last Δφ 값은 baseline과 달라야 함
    assert (
        changed["magnitude"],
        changed["severity"],
    ) != (
        base["magnitude"],
        base["severity"],
    )
