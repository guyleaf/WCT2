"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import os
import random
from pathlib import Path

import torch
import tqdm

from model import WaveDecoder, WaveEncoder
from utils.core import feature_wct
from utils.io import (
    Timer,
    collect_images,
    collect_images_from_images,
    compute_label_info,
    load_segment,
    open_image,
    save_image,
)


class WCT2:
    def __init__(
        self,
        model_path="./model_checkpoints",
        option_unpool="cat5",
        device="cuda:0",
        verbose=False,
    ):
        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(
                    model_path, "wave_encoder_{}_l4.pth".format(option_unpool)
                ),
                map_location=lambda storage, loc: storage,
            )
        )
        self.decoder.load_state_dict(
            torch.load(
                os.path.join(
                    model_path, "wave_decoder_{}_l4.pth".format(option_unpool)
                ),
                map_location=lambda storage, loc: storage,
            )
        )
        self.encoder.eval()
        self.decoder.eval()

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x, transfer_at):
        skips = {}
        feats = {"encoder": {}, "decoder": {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if "encoder" in transfer_at:
                feats["encoder"][level] = x

        if "encoder" not in transfer_at:
            feats["decoder"][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if "decoder" in transfer_at:
                feats["decoder"][level - 1] = x
        return feats, skips

    @torch.inference_mode()
    def transfer(
        self,
        content,
        style,
        content_segment,
        style_segment,
        alpha=1,
        transfer_at={"encoder", "skip", "decoder"},
    ) -> torch.Tensor:
        assert not (
            transfer_at - {"encoder", "decoder", "skip"}
        ), "invalid transfer_at: {}".format(transfer_at)
        assert transfer_at, "empty transfer_at"

        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style, transfer_at)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ["pool1", "pool2", "pool3"]

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if "encoder" in transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(
                    content_feat,
                    style_feats["encoder"][level],
                    content_segment,
                    style_segment,
                    label_set,
                    label_indicator,
                    alpha=alpha,
                    device=self.device,
                )
                self.print_("transfer at encoder {}".format(level))
        if "skip" in transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(
                        content_skips[skip_level][component],
                        style_skips[skip_level][component],
                        content_segment,
                        style_segment,
                        label_set,
                        label_indicator,
                        alpha=alpha,
                        device=self.device,
                    )
                self.print_("transfer at skip {}".format(skip_level))

        for level in [4, 3, 2, 1]:
            if (
                "decoder" in transfer_at
                and level in style_feats["decoder"]
                and level in wct2_dec_level
            ):
                content_feat = feature_wct(
                    content_feat,
                    style_feats["decoder"][level],
                    content_segment,
                    style_segment,
                    label_set,
                    label_indicator,
                    alpha=alpha,
                    device=self.device,
                )
                self.print_("transfer at decoder {}".format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ["encoder", None]:
        for d in ["decoder", None]:
            for s in ["skip", None]:
                _ret = set([e, d, s]) & set(["encoder", "decoder", "skip"])
                if _ret:
                    ret.append(_ret)
    return ret


def run_bulk(config):
    device = "cpu" if config.cpu or not torch.cuda.is_available() else "cuda:0"
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add("encoder")
    if config.transfer_at_decoder:
        transfer_at.add("decoder")
    if config.transfer_at_skip:
        transfer_at.add("skip")

    if config.option_mode == "p2p":
        content_fnames = collect_images(config.content)
        style_fnames = collect_images_from_images(
            content_fnames, config.content, config.style
        )
    elif config.option_mode == "random":
        content_fnames = collect_images(config.content)
        style_fnames = random.choices(
            collect_images(config.style), k=len(content_fnames)
        )
    else:
        raise NotImplementedError(f"Unsupported mode {config.option_mode}.")

    if config.content_segment and config.style_segment:
        content_segment_fnames = collect_images_from_images(
            content_fnames, config.content, config.content_segment
        )
        style_segment_fnames = collect_images_from_images(
            style_fnames, config.style, config.style_segment
        )
    else:
        content_segment_fnames = [None] * len(content_fnames)
        style_segment_fnames = [None] * len(style_fnames)

    wct2 = WCT2(
        option_unpool=config.option_unpool,
        device=device,
        verbose=config.verbose,
    )

    for _content, _style, _content_segment, _style_segment in tqdm.tqdm(
        zip(content_fnames, style_fnames, content_segment_fnames, style_segment_fnames)
    ):
        _rel = _content.relative_to(config.content)
        _output: Path = config.output / _rel

        content, unpadded_size = open_image(_content, config.image_size)
        content = content.to(device)
        style, _ = open_image(_style, config.image_size)
        style = style.to(device)

        content_segment = load_segment(_content_segment, config.image_size)
        style_segment = load_segment(_style_segment, config.image_size)

        if config.transfer_all:
            transfer_ats = get_all_transfer()
        else:
            transfer_ats = [transfer_at]

        for _transfer_at in transfer_ats:
            with Timer("Elapsed time in whole WCT: {}", config.verbose):
                if config.transfer_all:
                    postfix = "_".join(sorted(list(_transfer_at)))
                    fname_output = _output.with_name(
                        f"{_rel.stem}_{config.option_unpool}_{postfix}.png"
                    )
                else:
                    fname_output = _output.with_suffix(".png")

                print("------ transfer:", fname_output)
                img = wct2.transfer(
                    content,
                    style,
                    content_segment,
                    style_segment,
                    alpha=config.alpha,
                    transfer_at=_transfer_at,
                )
                save_image(img.squeeze(0), fname_output, unpadded_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--content", type=str, default="./examples/content")
    parser.add_argument("--content_segment", type=str, default=None)
    parser.add_argument("--style", type=str, default="./examples/style")
    parser.add_argument("--style_segment", type=str, default=None)
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument(
        "--option_unpool", type=str, default="cat5", choices=["sum", "cat5"]
    )
    parser.add_argument(
        "--option_mode", type=str, default="p2p", choices=["p2p", "random"]
    )
    parser.add_argument("-e", "--transfer_at_encoder", action="store_true")
    parser.add_argument("-d", "--transfer_at_decoder", action="store_true")
    parser.add_argument("-s", "--transfer_at_skip", action="store_true")
    parser.add_argument("-a", "--transfer_all", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    config = parser.parse_args()

    print(config)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    """
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    """
    run_bulk(config)
