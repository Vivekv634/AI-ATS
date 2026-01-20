"""
Base model classes for AI-ATS data models.

Provides common fields and functionality shared across all models.
"""

from datetime import datetime
from typing import Any, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2 compatibility with MongoDB."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(cls.validate),
                    ]
                ),
            ],
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, value: Any) -> ObjectId:
        """Validate and convert string to ObjectId."""
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str) and ObjectId.is_valid(value):
            return ObjectId(value)
        raise ValueError(f"Invalid ObjectId: {value}")


class TimestampMixin(BaseModel):
    """Mixin providing timestamp fields for models."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BaseDocument(TimestampMixin):
    """
    Base document model for MongoDB collections.

    Provides common fields and configuration for all database documents.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
        use_enum_values=True,
    )

    id: Optional[PyObjectId] = Field(default=None, alias="_id")

    def model_dump_mongo(self) -> dict[str, Any]:
        """Convert model to MongoDB-compatible dictionary."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if data.get("_id") is None:
            data.pop("_id", None)
        return data


class EmbeddedModel(BaseModel):
    """
    Base model for embedded documents (subdocuments).

    Use this for models that are embedded within other documents
    rather than stored in their own collection.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
    )
