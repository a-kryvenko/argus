"""Check this container's collector progress; no network or database required."""
import json
from clio.monitoring.heartbeat import check_heartbeat


def run(args):
    collector = args.collector
    if collector == 'worker':
        collectors = {name: check_heartbeat(name) for name in ('solar-wind', 'geomagnetic')}
        result = {'healthy': all(item['healthy'] for item in collectors.values()),
                  'collectors': collectors}
    else:
        result = check_heartbeat(collector)
    print(json.dumps(result))
    return 0 if result['healthy'] else 1
