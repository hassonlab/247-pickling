import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-type", type=str, default="glove")
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument(
        "--save-predictions", action="store_true", default=False
    )
    parser.add_argument(
        "--save-hidden-states", action="store_true", default=False
    )
    parser.add_argument("--subject", type=str, default="625")
    parser.add_argument("--conversation-id", type=int, default=0)
    parser.add_argument("--pkl-identifier", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--layer-idx", nargs="*", default=["all"])

    # whisper specific args
    parser.add_argument("--model-type", type=str, default="full")
    parser.add_argument("--shuffle-audio", type=str, default="none")
    parser.add_argument("--shuffle-words", type=str, default="none")
    parser.add_argument("--cutoff", type=float, default=0)
    parser.add_argument("--prod-comp-split", action="store_true", default=False)

    # mlm specific args
    parser.add_argument("--masked", action="store_true", default=False)
    parser.add_argument("--lctx", action="store_true", default=False)
    parser.add_argument("--rctx", action="store_true", default=False)
    parser.add_argument("--rctxp", action="store_true", default=False)
    parser.add_argument("--utt", action="store_true", default=False)


    # If running the code in debug mode
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = [
            "scripts/tfsemb_main.py",
            "--project-id",
            "podcast",
            "--pkl-identifier",
            "full",
            "--subject",
            "661",
            "--conversation-id",
            "1",
            "--embedding-type",
            "gpt2-xl",
            "--layer-idx",
            "last",
            "--context-length",
            "1024",
        ]

    args = parser.parse_args()

    return args
