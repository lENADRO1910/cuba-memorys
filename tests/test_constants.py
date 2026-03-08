"""Tests for constants.py — validation of thresholds and tool definitions.

Ensures domain constants are consistent with system invariants.
"""
import pytest

from cuba_memorys.constants import (
    CONTRADICTION_THRESHOLD,
    DEDUP_THRESHOLD,
    EXPORT_MAX_ENTITIES,
    EXPORT_MAX_ERRORS,
    EXPORT_MAX_OBSERVATIONS,
    EXPORT_MAX_RELATIONS,
    EXPORT_MAX_SESSIONS,
    GRAPH_RELATIONS_SQL,
    MAX_CONTENT_LENGTH,
    MAX_ERROR_MESSAGE_LENGTH,
    MAX_NAME_LENGTH,
    REM_IDLE_SECONDS,
    TOOL_DEFINITIONS,
    VALID_ENTITY_TYPES,
)


class TestThresholds:
    """Verify threshold invariants."""

    def test_dedup_greater_than_contradiction(self) -> None:
        """🟢 Dedup threshold > contradiction threshold (strict ordering).

        Rationale: If contradiction_threshold were >= dedup_threshold,
        contradictions would be classified as duplicates and silently
        discarded instead of triggering supersession.
        """
        assert DEDUP_THRESHOLD > CONTRADICTION_THRESHOLD

    def test_thresholds_in_unit_interval(self) -> None:
        """🟢 All similarity thresholds in [0, 1]."""
        for t in [DEDUP_THRESHOLD, CONTRADICTION_THRESHOLD]:
            assert 0.0 <= t <= 1.0

    def test_export_limits_positive(self) -> None:
        """🟢 All export limits are positive integers."""
        for limit in [
            EXPORT_MAX_ENTITIES,
            EXPORT_MAX_OBSERVATIONS,
            EXPORT_MAX_RELATIONS,
            EXPORT_MAX_ERRORS,
            EXPORT_MAX_SESSIONS,
        ]:
            assert isinstance(limit, int)
            assert limit > 0

    def test_input_length_limits(self) -> None:
        """🟢 Input validation limits are reasonable."""
        assert MAX_NAME_LENGTH > 0
        assert MAX_CONTENT_LENGTH > MAX_NAME_LENGTH
        assert MAX_ERROR_MESSAGE_LENGTH > 0

    def test_rem_idle_positive(self) -> None:
        """🟢 REM sleep idle time is positive."""
        assert REM_IDLE_SECONDS > 0


class TestValidEntityTypes:
    """Verify entity type enum."""

    def test_is_frozenset(self) -> None:
        """🟢 Immutable collection."""
        assert isinstance(VALID_ENTITY_TYPES, frozenset)

    def test_expected_types(self) -> None:
        """🟢 Contains all expected types."""
        expected = {"concept", "project", "technology", "person", "pattern", "config"}
        assert VALID_ENTITY_TYPES == expected

    def test_all_lowercase(self) -> None:
        """🟢 All types are lowercase."""
        for t in VALID_ENTITY_TYPES:
            assert t == t.lower()


class TestToolDefinitions:
    """Verify MCP tool schema consistency."""

    def test_12_tools_defined(self) -> None:
        """🟢 All 12 tools are defined."""
        assert len(TOOL_DEFINITIONS) == 12

    def test_unique_names(self) -> None:
        """🟢 All tool names are unique."""
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_all_have_required_fields(self) -> None:
        """🟢 Each tool has name, description, inputSchema."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_expected_tool_names(self) -> None:
        """🟢 All expected tool names present."""
        names = {t["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "cuba_alma", "cuba_cronica", "cuba_puente", "cuba_faro",
            "cuba_alarma", "cuba_remedio", "cuba_expediente", "cuba_jornada",
            "cuba_decreto", "cuba_zafra", "cuba_eco", "cuba_vigia",
        }
        assert names == expected

    def test_all_have_required_properties(self) -> None:
        """🟢 Each inputSchema has 'required' field."""
        for tool in TOOL_DEFINITIONS:
            schema = tool["inputSchema"]
            assert "required" in schema
            assert isinstance(schema["required"], list)
            assert len(schema["required"]) > 0

    def test_graph_relations_sql_has_join(self) -> None:
        """🟢 Graph SQL uses proper JOINs."""
        assert "JOIN brain_entities" in GRAPH_RELATIONS_SQL
        assert "FROM brain_relations" in GRAPH_RELATIONS_SQL
