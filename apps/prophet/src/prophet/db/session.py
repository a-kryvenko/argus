from common.db.session import get_db_session as _get_db_session


def get_db_session():
    return _get_db_session('prophet')
