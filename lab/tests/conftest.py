import pytest

from labhelpers import Env, build_client


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture
def client(env):
    c = build_client(env)
    with c:
        yield c
    c.app.state.get_store().db.close()
