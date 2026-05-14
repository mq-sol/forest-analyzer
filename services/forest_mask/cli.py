from datetime import datetime
from pathlib import Path

import click

from services.common.image import load_rgb, save_gray, save_mask
from services.forest_mask.exg import calc_exg
from services.forest_mask.mask import calc_forest_mask


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
def main(input_file: Path, threshold: float) -> None:
    rgb = load_rgb(input_file)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M%S")

    exg = calc_exg(rgb)

    exg_output_dir = Path("data/intermediate/exg")
    exg_output_dir.mkdir(parents=True, exist_ok=True)

    exg_path = exg_output_dir / f"exg_{formatted_time}.jpg"
    save_gray(exg, exg_path)
    click.echo(f"Saved: {exg_path}")

    forest_mask = calc_forest_mask(exg, threshold)

    mask_output_dir = Path("data/intermediate/forest_mask")
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_output_dir / f"fmask_{formatted_time}.jpg"
    save_mask(forest_mask, mask_path)
    click.echo(f"Saved: {mask_path}")


if __name__ == "__main__":
    main()
