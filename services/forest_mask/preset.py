from dataclasses import dataclass


@dataclass(frozen=True)
class ForestMaskPreset:
    threshold: float
    texture_kernel: int
    texture_th: float

    closing: int
    opening: int

    anomaly_closing: int
    anomaly_opening: int
    healthy_closing: int
    healthy_opening: int

    min_size: int
    anomaly_min_size: int

    sat_th: float
    lab_a_th: float
    anomaly_sat_th: float
    anomaly_val_th: float

    context_min_size: int
    context_max_size: int
    context_buffer_size: int
    context_healthy_ratio: float

    min_bbox_area: int
    max_shape_score: float | None


PRESETS: dict[str, ForestMaskPreset] = {
    "default": ForestMaskPreset(
        threshold=10.0,
        texture_kernel=5,
        texture_th=5.0,
        closing=5,
        opening=3,
        anomaly_closing=5,
        anomaly_opening=3,
        healthy_closing=7,
        healthy_opening=3,
        min_size=100,
        anomaly_min_size=300,
        sat_th=80.0,
        lab_a_th=125.0,
        anomaly_sat_th=60.0,
        anomaly_val_th=170.0,
        context_min_size=30,
        context_max_size=2000,
        context_buffer_size=15,
        context_healthy_ratio=0.4,
        min_bbox_area=100,
        max_shape_score=None,
    ),
    "sensitive": ForestMaskPreset(
        threshold=8.0,
        texture_kernel=5,
        texture_th=3.0,
        closing=5,
        opening=3,
        anomaly_closing=7,
        anomaly_opening=3,
        healthy_closing=7,
        healthy_opening=3,
        min_size=80,
        anomaly_min_size=150,
        sat_th=70.0,
        lab_a_th=128.0,
        anomaly_sat_th=80.0,
        anomaly_val_th=150.0,
        context_min_size=20,
        context_max_size=3000,
        context_buffer_size=21,
        context_healthy_ratio=0.3,
        min_bbox_area=50,
        max_shape_score=0.9,
    ),
    "strict": ForestMaskPreset(
        threshold=15.0,
        texture_kernel=5,
        texture_th=8.0,
        closing=5,
        opening=5,
        anomaly_closing=5,
        anomaly_opening=5,
        healthy_closing=7,
        healthy_opening=3,
        min_size=150,
        anomaly_min_size=500,
        sat_th=90.0,
        lab_a_th=122.0,
        anomaly_sat_th=50.0,
        anomaly_val_th=190.0,
        context_min_size=50,
        context_max_size=1200,
        context_buffer_size=15,
        context_healthy_ratio=0.5,
        min_bbox_area=150,
        max_shape_score=0.8,
    ),
}


def get_preset(name: str) -> ForestMaskPreset:
    return PRESETS[name]
