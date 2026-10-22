import sys
import types

from videox.doctor import _torch_status


def test_status_poll_does_not_initialize_cuda(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0"
    torch.version = types.SimpleNamespace(cuda="12.8")

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            raise AssertionError("status poll must not create a CUDA context")

    torch.cuda = Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _torch_status() == ("2.11.0", True)


def test_status_poll_reports_a_cpu_build(monkeypatch):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+cpu"
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _torch_status() == ("2.11.0+cpu", False)
