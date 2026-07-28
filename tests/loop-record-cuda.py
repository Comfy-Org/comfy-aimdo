import comfy_aimdo.control as control

assert control.init("cuda")

import torch


M = 1024 * 1024


assert torch.cuda.get_allocator_backend() == "cudaMallocAsync"
assert control.init_device(torch.cuda.current_device())
control.set_log_none()


def run_alias_iteration(stream, expected=None):
    first = torch.empty(M, dtype=torch.uint8, device="cuda")
    first.fill_(17)
    first_ptr = first.data_ptr()
    del first

    second = torch.empty(M, dtype=torch.uint8, device="cuda")
    assert torch.all(second == 17).item()
    second_ptr = second.data_ptr()
    del second

    pointers = first_ptr, second_ptr
    if expected is None:
        assert first_ptr != second_ptr
    else:
        assert pointers == expected
    return pointers


def test_graph_reuse(stream):
    control.push_record(stream)
    control.iterate()
    pointers = run_alias_iteration(stream)
    graph = control.pop()
    assert graph

    for _ in range(4):
        control.push_record(stream, graph)
        control.iterate()
        run_alias_iteration(stream, pointers)
        assert control.pop() == graph

    control.destroy_record(graph)
    print(f"graph replay: {pointers[0]:#x}, {pointers[1]:#x}")


def run_child(stream, expected=None):
    control.push_record(stream)
    pointer = expected
    for _ in range(4):
        control.iterate()
        value = torch.empty(M, dtype=torch.uint8, device="cuda")
        if pointer is None:
            pointer = value.data_ptr()
        else:
            assert value.data_ptr() == pointer
        del value
    assert control.pop() is None
    return pointer


def test_nested_graph_reuse(stream):
    control.push_record(stream)
    control.iterate()
    pointer = run_child(stream)
    graph = control.pop()

    for _ in range(3):
        control.push_record(stream, graph)
        control.iterate()
        run_child(stream, pointer)
        assert control.pop() == graph

    control.destroy_record(graph)
    print(f"nested graph replay: {pointer:#x}")


def test_passthrough(stream, other):
    external = torch.empty(M, dtype=torch.uint8, device="cuda")

    control.push_record(stream)
    control.iterate()
    del external
    with torch.cuda.stream(other):
        value = torch.empty(M, dtype=torch.uint8, device="cuda")
        value.fill_(91)
        del value
    graph = control.pop()
    other.synchronize()
    control.destroy_record(graph)
    print("external and other-stream passthrough: ok")


def test_mismatch_cleanup(stream):
    control.push_record(stream)
    control.iterate()
    value = torch.empty(M, dtype=torch.uint8, device="cuda")
    del value
    graph = control.pop()

    control.push_record(stream, graph)
    control.iterate()
    try:
        torch.empty(2 * M, dtype=torch.uint8, device="cuda")
    except RuntimeError:
        pass
    else:
        raise AssertionError("allocation graph mismatch was not reported")

    try:
        control.pop()
    except RuntimeError as error:
        assert error.graph == graph
    else:
        raise AssertionError("poisoned allocation graph pop did not fail")
    control.destroy_record(graph)
    print("mismatched graph cleanup: ok")


def test_cross_stream_use_is_rejected(stream, other):
    control.push_record(stream)
    control.iterate()
    value = torch.empty(M, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(other):
        other.wait_stream(stream)
        value.fill_(7)
    value.record_stream(other)
    del value
    torch.cuda.synchronize()

    try:
        control.pop()
    except RuntimeError as error:
        graph = error.graph
        assert "outside its record stream" in str(error)
    else:
        raise AssertionError("cross-stream allocation graph use was not rejected")
    control.destroy_record(graph)
    print("cross-stream graph rejection: ok")


stream = torch.cuda.Stream()
other = torch.cuda.Stream()
with torch.cuda.stream(stream):
    test_graph_reuse(stream)
    test_nested_graph_reuse(stream)
    test_passthrough(stream, other)
    test_mismatch_cleanup(stream)
    test_cross_stream_use_is_rejected(stream, other)

test_graph_reuse(torch.cuda.default_stream())

control.deinit()
print("PyTorch allocation graph tests passed")
