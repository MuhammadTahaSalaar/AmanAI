"""Tests for the data processing pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.base_processor import Document
from src.data_processing.json_processor import JSONProcessor


# ── Document Tests ───────────────────────────────────────────────────────────


class TestDocument:
    """Test the Document dataclass."""

    def test_creation(self):
        doc = Document(content="Hello", metadata={"source": "test"})
        assert doc.content == "Hello"
        assert doc.metadata == {"source": "test"}

    def test_to_dict(self):
        doc = Document(content="text", metadata={"key": "val"})
        d = doc.to_dict()
        assert d == {"content": "text", "metadata": {"key": "val"}}

    def test_from_dict(self):
        data = {"content": "text", "metadata": {"key": "val"}}
        doc = Document.from_dict(data)
        assert doc.content == "text"
        assert doc.metadata["key"] == "val"

    def test_roundtrip(self):
        original = Document(content="round trip", metadata={"a": 1})
        restored = Document.from_dict(original.to_dict())
        assert original.content == restored.content
        assert original.metadata == restored.metadata

    def test_default_metadata(self):
        doc = Document(content="no meta")
        assert doc.metadata == {}


# ── JSON Processor Tests ─────────────────────────────────────────────────────


class TestJSONProcessor:
    """Test the JSON FAQ file processor."""

    def _write_json(self, tmp_dir: Path, data: dict) -> Path:
        path = tmp_dir / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_single_category(self, tmp_path):
        data = {
            "categories": [
                {
                    "category": "Transfers",
                    "questions": [
                        {
                            "question": "How do I transfer?",
                            "answer": "Use the app.",
                        }
                    ],
                }
            ]
        }
        path = self._write_json(tmp_path, data)
        processor = JSONProcessor(file_path=path)
        docs = processor.process()
        assert len(docs) == 1
        assert "How do I transfer?" in docs[0].content
        assert "Use the app." in docs[0].content
        assert docs[0].metadata["category"] == "Transfers"

    def test_empty_categories(self, tmp_path):
        data = {"categories": []}
        path = self._write_json(tmp_path, data)
        processor = JSONProcessor(file_path=path)
        docs = processor.process()
        assert docs == []

    def test_multiple_questions(self, tmp_path):
        data = {
            "categories": [
                {
                    "category": "Accounts",
                    "questions": [
                        {"question": "Q1?", "answer": "A1"},
                        {"question": "Q2?", "answer": "A2"},
                    ],
                }
            ]
        }
        path = self._write_json(tmp_path, data)
        processor = JSONProcessor(file_path=path)
        docs = processor.process()
        assert len(docs) == 2
