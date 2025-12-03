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
