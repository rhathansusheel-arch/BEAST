from broker.order_executor import OrderExecutor


def test_order_executor_instantiates():
    executor = OrderExecutor(client=None)
    assert executor is not None
