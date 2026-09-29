from pathlib import Path


def run(remaining):
    from common.config import get_config
    from alembic.config import Config, CommandLine
    get_config()
    cli = CommandLine(prog='prophet migrate')
    if not remaining:
        cli.parser.print_help()
        return
    options = cli.parser.parse_args(remaining)
    config = Config()
    config.cmd_opts = options
    config.set_main_option('script_location', str(Path(__file__).resolve().parents[1] / 'migrations'))
    cli.run_cmd(config, options)
