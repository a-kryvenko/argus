"""Run Alembic using migrations included in the Clio package."""
from pathlib import Path


def run(remaining):
    from alembic.config import CommandLine, Config
    cli = CommandLine(prog='clio migrate')
    if not remaining:
        cli.parser.print_help()
        return
    options = cli.parser.parse_args(remaining)
    config = Config()
    config.cmd_opts = options
    config.set_main_option('script_location', str(Path(__file__).resolve().parents[1] / 'db' / 'migrations'))
    cli.run_cmd(config, options)
