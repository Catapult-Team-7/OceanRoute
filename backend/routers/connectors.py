from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from connectors import DataConnectorService
from db.database import get_repo
from db.demo_data import DemoOceanRepository

router = APIRouter()


def get_connector_service(repo: Annotated[DemoOceanRepository, Depends(get_repo)]):
    return DataConnectorService(repo.trainer)


@router.get("/connectors")
async def list_connectors(service: Annotated[DataConnectorService, Depends(get_connector_service)]):
    return {"connectors": service.list_connectors()}


@router.post("/connectors/{connector_id}/preview")
async def preview_connector(
    connector_id: str,
    service: Annotated[DataConnectorService, Depends(get_connector_service)],
):
    preview = service.preview(connector_id)
    if preview.get("status") == "error" and "Unknown connector" in preview.get("message", ""):
        raise HTTPException(status_code=404, detail=preview["message"])
    return preview


@router.post("/connectors/{connector_id}/sync")
async def sync_connector(
    connector_id: str,
    service: Annotated[DataConnectorService, Depends(get_connector_service)],
):
    result = service.sync(connector_id)
    if result.get("status") == "error" and "not implemented" in result.get("message", ""):
        raise HTTPException(status_code=404, detail=result["message"])
    return result
