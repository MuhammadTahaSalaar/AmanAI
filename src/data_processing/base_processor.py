"""Abstract base class for all data processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    """A processed document chunk ready for embedding.

    Attributes:
        content: The text content of the chunk.
        metadata: Key-value metadata (product, source, category, etc.).
    """

    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {"content": self.content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Deserialize from a plain dictionary."""
        return cls(content=data["content"], metadata=data.get("metadata", {}))


class BaseProcessor(ABC):
    """Abstract base class that all data processors must implement."""

    @abstractmethod
    def process(self) -> list[Document]:
        """Process the data source and return a list of Document objects.

        Returns:
            List of Document objects with content and metadata.
        """
