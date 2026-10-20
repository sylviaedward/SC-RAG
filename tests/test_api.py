import unittest

from sc_rag.api import app


class ApiContractTests(unittest.TestCase):
    def test_expected_routes_are_registered(self) -> None:
        routes = {route.path for route in app.routes}
        self.assertTrue({"/", "/docs", "/health", "/ready", "/search", "/answer", "/literature-review"} <= routes)

