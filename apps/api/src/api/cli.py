"""API maintenance commands."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('migrate', add_help=False)
    args, remaining = parser.parse_known_args(argv)
    from api.commands.migrate import run
    return run(remaining)
