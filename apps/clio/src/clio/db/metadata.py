"""Register all Clio tables for migration metadata."""
from clio.db.base import Base
from clio.db.models import measurement
from clio.db.models import normalized_observation
from clio.db.models import observation_source_status
from clio.db.models import scheduled_job
from clio.db.models import gong_snapshot
from clio.db.models import goes_snapshot
from clio.db.models import donki_poll
from clio.db.models import enlil_profile
