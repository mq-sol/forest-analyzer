from datetime import datetime
from pathlib import Path

import click

from services.common.image import load_rgb, save_gray, save_mask
from services.forest_mask.exg import calc_exg
from services.forest_mask.mask import calc_forest_mask
from services.forest_mask.morphology import (
    apply_morphology,
    apply_closing,
    apply_opening,
)

from services.forest_mask.components import remove_small_components
from services.forest_mask.texture import calc_texture


@click.command()
@click.argument(
    "input_file",
    type=click.Path(
        exists=True,
        path_type=Path,
    ),
)
@click.option(
    "--threshold",
    type=float,
    default=10.0,
    show_default=True,
    help="ExG threshold for forest mask",
)
@click.option(
    "--texture-kernel",
    type=int,
    default=5,
    show_default=True,
    help="Kernel size for local texture (variance)",
)
@click.option(
    "--closing",
    type=int,
    default=5,
    show_default=True,
    help="Kernel size for closing",
)
@click.option(
    "--opening",
    type=int,
    default=3,
    show_default=True,
    help="Kernel size for opening",
)
def main(
    input_file: Path,
    threshold: float,
    texture_kernel: int,
    closing: int,
    opening: int,
) -> None:
    rgb = load_rgb(input_file)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M%S")

    exg = calc_exg(rgb)

    exg_output_dir = Path("data/intermediate/exg")
    exg_output_dir.mkdir(parents=True, exist_ok=True)

    exg_path = exg_output_dir / f"exg_{formatted_time}.jpg"
    save_gray(exg, exg_path)
    click.echo(f"Saved: {exg_path}")

    texture = calc_texture(exg, texture_kernel)

    texture_path = exg_output_dir / f"texture_{formatted_time}.jpg"
    save_gray(texture, texture_path)
    click.echo(f"Saved: {texture_path}")

    forest_mask = calc_forest_mask(texture, threshold)

    mask_output_dir = Path("data/intermediate/forest_mask")
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_output_dir / f"fmask_{formatted_time}.jpg"
    save_mask(forest_mask, mask_path)
    click.echo(f"Saved: {mask_path}")

    # morph_cl_mask = apply_closing(forest_mask, closing)
    # morph_cl_mask_path = mask_output_dir / f"fmask_morph_cl_cl_{formatted_time}.jpg"
    # save_mask(morph_cl_mask, morph_cl_mask_path)
    # click.echo(f"Saved: {morph_cl_mask_path}")

    # morph_op_mask = apply_opening(morph_cl_mask, opening)
    # morph_op_mask_path = mask_output_dir / f"fmask_morph_op_cl_{formatted_time}.jpg"
    # save_mask(morph_op_mask, morph_op_mask_path)
    # click.echo(f"Saved: {morph_op_mask_path}")

    morph_mask = apply_morphology(forest_mask, closing, opening)
    morph_mask_path = mask_output_dir / f"fmask_morph_{formatted_time}.jpg"
    save_mask(morph_mask, morph_mask_path)
    click.echo(f"Saved: {morph_mask_path}")

    compo_mask = remove_small_components(morph_mask, 100)
    compo_mask_path = mask_output_dir / f"compo_{formatted_time}.jpg"
    save_mask(compo_mask, compo_mask_path)
    click.echo(f"Saved: {compo_mask_path}")


if __name__ == "__main__":
    main()
