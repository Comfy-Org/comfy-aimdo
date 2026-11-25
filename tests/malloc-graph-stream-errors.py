from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from comfy_aimdo import control
from comfy_aimdo.malloc_graph import MallocGraph


class StreamErrors(unittest.TestCase):
    def setUp(self):
        self.previous = SimpleNamespace(cuda_stream=1)
        self.stream = SimpleNamespace(cuda_stream=2)
        self.graph = MallocGraph(1, self.previous)
        self.set_stream = Mock(return_value=True)
        self.library = SimpleNamespace(
            malloc_graph_set_stream=self.set_stream,
            malloc_graph_destroy=Mock(),
        )
        self.patch = patch.object(control, "lib", self.library)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_restore_on_success(self):
        with self.graph.use_stream(self.stream):
            self.assertIs(self.graph._stream, self.stream)
        self.assertIs(self.graph._stream, self.previous)
        self.assertEqual([call.args[1].value for call in self.set_stream.call_args_list], [2, 1])

    def test_entry_failure(self):
        self.set_stream.return_value = False
        with self.assertRaisesRegex(RuntimeError, "aimdo memory compile error"):
            with self.graph.use_stream(self.stream):
                self.fail("A rejected stream must not be used")
        self.assertIs(self.graph._stream, self.previous)
        self.assertEqual(self.set_stream.call_count, 1)

    def test_restore_failure_without_body_error(self):
        self.set_stream.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, "aimdo memory compile error"):
            with self.graph.use_stream(self.stream):
                pass
        self.assertIs(self.graph._stream, self.stream)

    def test_preserve_body_exception(self):
        for error in (MemoryError("allocation failed"), ValueError("invalid input"), KeyboardInterrupt()):
            for restored in (True, False):
                with self.subTest(error=type(error).__name__, restored=restored):
                    self.graph._stream = self.previous
                    self.set_stream.reset_mock()
                    self.set_stream.side_effect = [True, restored]
                    with self.assertRaises(type(error)) as caught:
                        with self.graph.use_stream(self.stream):
                            raise error
                    self.assertIs(caught.exception, error)
                    self.assertEqual(self.set_stream.call_count, 2)
                    self.assertIs(self.graph._stream, self.previous if restored else self.stream)

    def test_nested_restore_failures(self):
        nested = SimpleNamespace(cuda_stream=3)
        self.set_stream.side_effect = [True, True, False, False]
        error = MemoryError("allocation failed")
        with self.assertRaises(MemoryError) as caught:
            with self.graph.use_stream(self.stream):
                with self.graph.use_stream(nested):
                    raise error
        self.assertIs(caught.exception, error)
        self.assertEqual([call.args[1].value for call in self.set_stream.call_args_list], [2, 3, 2, 1])
        self.assertIs(self.graph._stream, nested)


if __name__ == "__main__":
    unittest.main()
