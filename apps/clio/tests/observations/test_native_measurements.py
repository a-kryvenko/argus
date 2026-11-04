import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pandas as pd
from sqlalchemy.dialects import postgresql

from clio.ingestion.utils.adapters import wide_to_measurements
from clio.ingestion.utils.measurements import store_native

NOW = datetime(2026, 10, 3, tzinfo=UTC)


def test_dataset_conversion_keeps_independent_values_and_solar_means():
    frame = pd.DataFrame({'timestamp': [NOW], 'bz': [-2.], 'bx': [float('nan')],
                          'f10_7_81mean': [110.], 'bt': [3.]})
    rows = wide_to_measurements(frame)
    assert dict(zip(rows.metric, rows.value)) == {'bz': -2., 'f10_7_81mean': 110., 'bt': 3.}
    assert rows.observed_at.eq(NOW).all()


def test_receipt_alone_is_not_a_revision():
    session = AsyncMock()
    asyncio.run(store_native(session, [dict(metric='bz', value=-2., observed_at=NOW,
        received_at=NOW, source_product='swpc.rtsw_mag')]))
    query = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    where = query.split(' WHERE ')[1]
    assert 'value IS DISTINCT FROM' in where
    assert 'received_at IS DISTINCT FROM' not in where
    assert 'quality' not in query
    assert 'ON CONFLICT ON CONSTRAINT uq_measurement_metric' in query
