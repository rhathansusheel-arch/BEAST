from core.risk_manager import RiskManager


def test_risk_manager_instantiates():
    manager = RiskManager(config={})
    assert manager is not None
