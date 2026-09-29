import json


def run(args):
    from common.config import get_config
    get_config()
    if args.command == 'status':
        from argus_prophet.services.releases.status import product_status
        from argus_prophet.services.generation.products import select_products
        result = ([product_status(product).model_dump(mode='json') for product in select_products('all')]
                  if args.product == 'all' else product_status(args.product).model_dump(mode='json'))
    elif args.command == 'slots':
        from argus_prophet.scheduling.jobs import list_slots
        result = list_slots(args.limit)
    else:
        from argus_prophet.services.runs import list_runs, describe_run
        result = list_runs(args.limit) if args.command == 'runs' else describe_run(args.run_id, args.inputs)
    print(json.dumps(result, default=str, indent=2))
