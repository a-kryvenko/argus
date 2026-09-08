"""Check this container's collector progress; no network or database required."""
import argparse
import json
from app.services.collector_heartbeat import check_heartbeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collector', choices=['solar-wind', 'geomagnetic'])
    result = check_heartbeat(parser.parse_args().collector)
    print(json.dumps(result))
    raise SystemExit(0 if result['healthy'] else 1)


if __name__ == '__main__':
    main()
