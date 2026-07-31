import argparse
import time
from pathlib import Path

import torch
from scipy.io import savemat

from dataset import reader_testdata
from network import CSSL
from src.utils import TensorToImage, check_and_make, save_image
from train import infer_ms_channels, infer_num_classes


def parse_bool_list(values):
    if isinstance(values, list):
        return values
    return [value.strip().lower() in {"1", "true", "yes", "fr"} for value in values.split(",")]


def load_model(args, device):
    ms_channels = args.ms_channels or infer_ms_channels(args.sensor)
    num_classes = args.num_classes or infer_num_classes(args.sensor)
    model = CSSL.load_from_checkpoint(
        args.checkpoint,
        ms_channels=ms_channels,
        num_classes=num_classes,
        task=args.task,
        sensor=args.sensor,
        map_location=device,
    )
    model.eval()
    return model.to(device)


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = load_model(args, device)
    fr_tests = parse_bool_list(args.fr_test)

    for fr_test in fr_tests:
        split = "FR" if fr_test else "RR"
        data_dir = Path(args.data_dir) / args.sensor / "Test" / split
        mat_files = sorted((data_dir / "MS").glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {data_dir / 'MS'}")

        start = time.time()
        with torch.no_grad():
            for ms_path in mat_files:
                ms, upms, pan, _ = reader_testdata(data_dir, ms_path.name, sensor=args.sensor, fr_test=fr_test)
                ms = ms.to(device)
                upms = upms.to(device)
                pan = pan.to(device)
                output = model(ms.unsqueeze(0), upms.unsqueeze(0), pan.unsqueeze(0))
                output_image = TensorToImage(output["fused"].squeeze(0))

                save_dir = Path(args.save_dir) / args.sensor / split
                save_mat = save_dir / "mat"
                save_tif = save_dir / "tif"
                check_and_make(save_mat)
                check_and_make(save_tif)

                savemat(save_mat / ms_path.name, {"hrms_image": output_image})
                bands = [2, 1, 0] if output_image.shape[-1] == 4 else [4, 2, 1]
                save_image(output_image, save_tif / ms_path.with_suffix(".png").name, bands)

        elapsed = (time.time() - start) / len(mat_files)
        print(f"{split} inference finished. Average running time: {elapsed:.4f}s")


def build_parser():
    parser = argparse.ArgumentParser(description="Run CSSL inference.")
    parser.add_argument("--data-dir", default="data/psc")
    parser.add_argument("--sensor", default="worldview-2")
    parser.add_argument("--checkpoint", default="checkpoints/CSSL-worldview-2.ckpt")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--fr-test", default="true")
    parser.add_argument("--ms-channels", default=None, type=int)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
