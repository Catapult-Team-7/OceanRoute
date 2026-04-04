from fastapi import Request

from .demo_data import DemoOceanRepository


def get_repo(request: Request) -> DemoOceanRepository:
    return request.app.state.repo
