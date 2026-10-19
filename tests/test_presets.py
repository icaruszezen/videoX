from videox.hardware.gpus import GpuInfo, batch_size_for_devices, batch_size_for_memory_mib, default_devices


def test_twelve_gigabyte_cards_use_five_frames():
    assert batch_size_for_memory_mib(12288) == 5
    assert batch_size_for_memory_mib(15 * 1024 - 1) == 5


def test_sixteen_gigabyte_and_above_use_nine_frames():
    assert batch_size_for_memory_mib(16376) == 9
    assert batch_size_for_memory_mib(20480) == 9


def test_two_gpus_default_to_both_devices_and_the_smaller_card():
    gpus = [
        GpuInfo(0, "RTX 4080", 16376),
        GpuInfo(1, "RTX 3080", 20480),
    ]
    devices = default_devices(gpus)
    assert devices == [0, 1]
    assert batch_size_for_devices(devices, gpus) == 9


def test_single_gpu_stays_on_that_card():
    gpus = [GpuInfo(1, "RTX 3080", 12288)]
    assert default_devices(gpus) == [1]
    assert batch_size_for_devices([1], gpus) == 5
