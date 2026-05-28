from datetime import datetime
from pathlib import Path

import click

from services.common.image import load_rgb, save_gray, save_mask, save_rgb
from services.forest_mask.exg import calc_exg
from services.forest_mask.mask import calc_forest_mask
from services.forest_mask.morphology import apply_morphology
from services.forest_mask.components import remove_small_components
from services.forest_mask.texture import calc_texture
from services.forest_mask.color import calc_hsv, calc_lab
from services.forest_mask.classifier import classify_forest_status
from services.forest_mask.overlay import create_mask_overlay
from services.forest_mask.context import filter_components_by_surrounding_healthy


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--threshold", type=float, default=10.0, show_default=True)
@click.option("--texture-kernel", type=int, default=5, show_default=True)
@click.option("--closing", type=int, default=5, show_default=True)
@click.option("--opening", type=int, default=3, show_default=True)
@click.option("--min-size", type=int, default=100, show_default=True)
@click.option("--sat-th", type=float, default=80.0, show_default=True)
@click.option("--lab-a-th", type=float, default=125.0, show_default=True)
@click.option("--anomaly-sat-th", type=float, default=60.0, show_default=True)
@click.option("--anomaly-val-th", type=float, default=170.0, show_default=True)
def main(
    input_file: Path,
    threshold: float,
    texture_kernel: int,
    closing: int,
    opening: int,
    min_size: int,
    sat_th: float,
    lab_a_th: float,
    anomaly_sat_th: float,
    anomaly_val_th: float,
) -> None:
    rgb = load_rgb(input_file)
    formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")

    feature_dir = Path("data/intermediate/features")
    mask_dir = Path("data/intermediate/forest_mask")
    class_dir = Path("data/intermediate/classifier")
    overlay_dir = Path("data/intermediate/overlay")

    feature_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    class_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # 特徴量生成
    exg = calc_exg(rgb)
    save_gray(exg, feature_dir / f"exg_{formatted_time}.jpg")

    hsv_h, hsv_s, hsv_v = calc_hsv(rgb)
    save_gray(hsv_h, feature_dir / f"hsv_h_{formatted_time}.jpg")
    save_gray(hsv_s, feature_dir / f"hsv_s_{formatted_time}.jpg")
    save_gray(hsv_v, feature_dir / f"hsv_v_{formatted_time}.jpg")

    lab_l, lab_a, lab_b = calc_lab(rgb)
    save_gray(lab_l, feature_dir / f"lab_l_{formatted_time}.jpg")
    save_gray(lab_a, feature_dir / f"lab_a_{formatted_time}.jpg")
    save_gray(lab_b, feature_dir / f"lab_b_{formatted_time}.jpg")

    texture = calc_texture(exg, texture_kernel)
    save_gray(texture, feature_dir / f"texture_{formatted_time}.jpg")

    click.echo(f"Saved features: {feature_dir}")

    # 基本森林マスク
    forest_mask = calc_forest_mask(exg, threshold)

    raw_mask_path = mask_dir / f"fmask_raw_{formatted_time}.jpg"
    save_mask(forest_mask, raw_mask_path)

    morph_mask = apply_morphology(forest_mask, closing, opening)
    morph_mask_path = mask_dir / f"fmask_morph_{formatted_time}.jpg"
    save_mask(morph_mask, morph_mask_path)

    compo_mask = remove_small_components(morph_mask, min_size)
    compo_mask_path = mask_dir / f"fmask_compo_{formatted_time}.jpg"
    save_mask(compo_mask, compo_mask_path)

    click.echo(f"Saved forest masks: {mask_dir}")

    # 正常 / 異常 / 不明分類
    healthy_mask, anomaly_mask, unknown_mask = classify_forest_status(
        exg=exg,
        s=hsv_s,
        v=hsv_v,
        lab_a=lab_a,
        texture=texture,
        exg_threshold=threshold,
        saturation_threshold=sat_th,
        lab_a_threshold=lab_a_th,
        anomaly_saturation_threshold=anomaly_sat_th,
        anomaly_value_threshold=anomaly_val_th,
        texture_threshold=texture_kernel,
    )

    save_mask(healthy_mask, class_dir / f"healthy_{formatted_time}.jpg")
    save_mask(anomaly_mask, class_dir / f"anomaly_{formatted_time}.jpg")
    save_mask(unknown_mask, class_dir / f"unknown_{formatted_time}.jpg")

    closing = 5
    opening = 3

    anomaly_morph = apply_morphology(anomaly_mask, closing, opening)
    anomaly_morph_path = mask_dir / f"amask_morph_{formatted_time}.jpg"
    save_mask(anomaly_morph, anomaly_morph_path)

    closing = 7
    opening = 3

    healthy_morph = apply_morphology(
        healthy_mask,
        closing,
        opening,
    )
    healthy_morph_path = mask_dir / f"hmask_morph_{formatted_time}.jpg"
    save_mask(healthy_morph, healthy_morph_path)

    click.echo(f"Saved classifier masks: {class_dir}")

    anomaly_compo = remove_small_components(
        anomaly_morph,
        min_size=300,
    )

    anomaly_compo_path = mask_dir / f"amask_compo_{formatted_time}.jpg"
    save_mask(anomaly_compo, anomaly_compo_path)

    anomaly_context_path = mask_dir / f"filter_context_{formatted_time}.png"
    anomaly_context = filter_components_by_surrounding_healthy(
        anomaly_mask=anomaly_compo,
        healthy_mask=healthy_morph,
        min_size=30,
        max_size=2000,
        buffer_size=15,
        min_healthy_ratio=0.4,
    )

    save_mask(anomaly_context, anomaly_context_path)

    anomaly_overlay_path = overlay_dir / f"an_overlay{formatted_time}.jpg"
    anomaly_overlay = create_mask_overlay(
        rgb,
        anomaly_context,
        color=(255, 0, 0),
        alpha=0.4,
    )
    save_rgb(anomaly_overlay, anomaly_overlay_path)


if __name__ == "__main__":
    main()
