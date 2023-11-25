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
