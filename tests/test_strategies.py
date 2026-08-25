from core.regime_strategies import RegimeStrategy


def test_regime_strategy_instantiates():
    strategy = RegimeStrategy(config={})
    assert strategy is not None
