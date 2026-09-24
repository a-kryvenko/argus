"""Check this container's collector progress; no network or database required."""
import argparse
import json
from argus_clio.services.collection.heartbeat import check_heartbeat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collector', choices=['solar-wind', 'geomagnetic', 'worker'])
    collector = parser.parse_args(argv).collector
    if collector == 'worker':
        collectors = {name: check_heartbeat(name) for name in ('solar-wind', 'geomagnetic')}
        result = {'healthy': all(item['healthy'] for item in collectors.values()),
                  'collectors': collectors}
    else:
        result = check_heartbeat(collector)
    print(json.dumps(result))
    raise SystemExit(0 if result['healthy'] else 1)


if __name__ == '__main__':
    main()
