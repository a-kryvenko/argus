"""Collection cadence shared by the scheduler and its diagnostics."""
COLLECTING_SCHEDULES = [
    (["v", "n", "t"], "*"),
    (["bx", "by", "bz"], "*"),
    (["kp", "ap"], "*"),
    (["dst", "sdo"], "*/5"),
    (["donki", "enlil", "sdo-browse"], "*/15"),
    (["f10_7", "f10_7_81mean", "s10_81mean", "m10_81mean", "y10_81mean", "goes", "gong"], "0"),
]


def poll_seconds(metric: str) -> int:
    for metrics, minute in COLLECTING_SCHEDULES:
        if metric in metrics:
            if minute == '*':
                return 60
            if minute.startswith('*/'):
                return int(minute[2:]) * 60
            return 3600
    raise ValueError(f'No collection schedule for {metric}')
