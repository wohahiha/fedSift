"""Keep declared package exports importable after API changes."""
import importlib
import pkgutil
import unittest

import fedsift


class PublicApiTests(unittest.TestCase):
    def test_declared_exports_resolve(self):
        for entry in pkgutil.iter_modules(fedsift.__path__):
            module = importlib.import_module(f"fedsift.{entry.name}")
            names = getattr(module, "__all__", ())
            with self.subTest(module=module.__name__):
                self.assertEqual(len(names), len(set(names)))
                for name in names:
                    self.assertTrue(hasattr(module, name), f"{module.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
