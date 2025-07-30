import numpy as np
from akula.graphite_models import GraphiteLCAModel, GraphiteTEAModel

class DummyLCA:
    def graphite_lca_single(self, scenario, slate, allocation, methods):
        return np.arange(len(methods))

class DummyTEA:
    def run(self, **kwargs):
        return {"msp_usd_per_kg": 1.23}

def test_graphite_wrappers():
    lca = GraphiteLCAModel(DummyLCA(), [("a", "b", "c")])
    scores = lca.single_score("s1", (0.3, 0.3, 0.4))
    assert np.all(scores == np.array([0]))

    tea = GraphiteTEAModel(DummyTEA())
    res = tea.run()
    assert res["msp_usd_per_kg"] == 1.23
