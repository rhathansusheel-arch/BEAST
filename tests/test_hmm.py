from core.hmm_engine import HMMEngine


def test_hmm_engine_instantiates():
    engine = HMMEngine(config={})
    assert engine is not None
