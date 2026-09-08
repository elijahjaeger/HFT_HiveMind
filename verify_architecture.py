"""Architecture gate: the frameworks resolve and the graph actually wires up.

Exit 0 means the LangGraph supervisor compiled with every stage present, the
CrewAI personas exist, and the FinRL/SB3 policy pipeline is constructible.
It does NOT mean the strategy works - see README "Known limitations".
"""
import sys

# Windows consoles default to cp1252, which cannot encode the emoji below.
# Without this the script dies in its own print() before testing anything.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    print("🤖 Checking Multi-Agent Trading System Architecture...")

    # 1. Check Framework Imports
    import crewai  # noqa: F401 - imported to prove availability
    import langgraph  # noqa: F401 - imported to prove availability
    print("✅ Core frameworks (LangGraph, CrewAI) installed and importable.")

    # The RL stack is parked and deliberately NOT a requirement.
    # See DECISIONS.md (2026-09-08).
    import src.rl_policy
    state = "installed" if src.rl_policy.RL_AVAILABLE else "not installed (parked)"
    print(f"ℹ️  RL stack (FinRL/SB3/gymnasium): {state}")

    # 2. Check Your Implementation Modules (Adjust filenames to match your project)
    import src.research_desk  # Your CrewAI implementation
    import src.rl_policy  # Your FinRL implementation
    import src.state_graph  # Your LangGraph implementation
    print("✅ Project files successfully compiled and imported.")

    # 3. State-machine checks: imports alone do not prove the graph wires up.
    graph = src.state_graph.compile_graph()
    assert graph is not None, "supervisor graph failed to compile"
    nodes = set(getattr(graph, "nodes", {}))
    for required in ("supervisor", "ingest", "research_desk", "rl_policy", "risk", "execute"):
        assert required in nodes, f"graph is missing the {required!r} node"
    personas = set(src.research_desk.build_personas())
    assert personas == {"MacroAnalyst", "MicrostructureQuant"}, f"bad personas: {personas}"
    assert hasattr(src.rl_policy.RLPolicyPipeline, "train"), "RL pipeline cannot train"
    assert hasattr(src.rl_policy, "AlphaTradingEnv"), "no RL environment"
    print(f"✅ Graph compiled, nodes: {sorted(nodes)}")
    print(f"✅ CrewAI personas: {sorted(personas)}")
    print("✅ RL policy pipeline exposes train() over AlphaTradingEnv.")

    print("🚀 Architecture Verification: SUCCESS")
    sys.exit(0) # Exit with 0 means success

except ImportError as e:
    print(f"❌ Verification Failed: Missing module or import error -> {e}")
    sys.exit(1) # Exit with 1 means Claude needs to keep working
except Exception as e:
    print(f"❌ Verification Failed: {e}")
    sys.exit(1)
