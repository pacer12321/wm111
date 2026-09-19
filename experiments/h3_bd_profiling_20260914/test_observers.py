"""CPU-only observer-contract tests; no torch/NPU imports required."""
import contextlib
import sys
import types
import unittest
from unittest.mock import patch

import profile_hooks as h


class ObserverTests(unittest.TestCase):
    def tearDown(self):
        h._ACTIVE = h._EVENTS = False
        h._STACK.clear()
        h._ROWS.clear()

    def test_inactive_preserves_identity_and_one_call(self):
        values = []
        obj = object()
        def fn(x, *, y):
            values.append((x, y))
            return obj
        wrapped = h.observer(fn, 'unit')
        self.assertIs(wrapped(1, y=2), obj)
        self.assertEqual(values, [(1, 2)])

    def test_inactive_preserves_exception(self):
        def fn():
            raise ValueError('sentinel')
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            h.observer(fn, 'unit')()

    def test_event_nesting_and_cleanup(self):
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): pass
        fake = types.SimpleNamespace(npu=types.SimpleNamespace(Event=Event),
                                     profiler=types.SimpleNamespace(record_function=lambda _: contextlib.nullcontext()))
        h._ACTIVE = h._EVENTS = True
        with patch.dict(sys.modules, torch=fake):
            with h.span('parent'):
                with h.span('child'):
                    pass
        self.assertEqual([(r['id'], r['parent']) for r in h._ROWS], [(0, None), (1, 0)])
        self.assertEqual(h._STACK, [])

    def test_active_does_not_swallow_exception(self):
        fake = types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=lambda _: contextlib.nullcontext()))
        h._ACTIVE = True
        with patch.dict(sys.modules, torch=fake):
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                with h.span('test'):
                    raise RuntimeError('sentinel')


if __name__ == '__main__':
    unittest.main()
