#!/usr/bin/env python3
"""
Pydantic models for API request/response validation.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class IndividualData(BaseModel):
    id: str = Field(..., description="Unique identifier for the individual")
    class_name: str = Field(..., alias="class", description="OWL class name")
    data_properties: Optional[Dict[str, Any]] = Field(default_factory=dict)
    object_properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class IndividualUpdate(BaseModel):
    data_properties: Optional[Dict[str, Any]] = Field(default=None)
    object_properties: Optional[Dict[str, Any]] = Field(default=None)


class StatusResponse(BaseModel):
    status: str
    ontology: Optional[str] = None
    individuals_count: Optional[int] = None
    classes_count: Optional[int] = None
    individuals: Optional[List[str]] = None
    message: Optional[str] = None


class OperationResponse(BaseModel):
    status: str
    id: Optional[str] = None
    message: Optional[str] = None
    individuals: Optional[int] = None
    relationships: Optional[int] = None
    added: Optional[int] = None
    failed: Optional[int] = None


class BatchIndividualsData(BaseModel):
    individuals: List[IndividualData] = Field(..., description="List of individuals to add")


class SemanticQuery(BaseModel):
    query: str
    top_k: int = 5


class CypherQuery(BaseModel):
    query: str


class ObjectQuery(BaseModel):
    object_id: str


class PositionQuery(BaseModel):
    object_id: str


class RoomQuery(BaseModel):
    object_id: str


class CategoryQuery(BaseModel):
    category: str
    room_id: Optional[str] = None


class AffordanceQuery(BaseModel):
    affordance: str
    room_id: Optional[str | int] = None
    top_k: int = 5


class LocateSemanticQuery(BaseModel):
    query: str
    room_id: Optional[str] = None
    top_k: int = 5


class SceneQuery(BaseModel):
    scene_id: str


class ClassQuery(BaseModel):
    class_name: str
