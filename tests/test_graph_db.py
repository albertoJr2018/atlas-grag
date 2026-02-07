"""
Tests for Neo4j graph database manager.
"""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.database.graph_db import GraphDatabaseManager, GraphDatabaseError


class TestGraphDatabaseManagerInitialization:
    """Tests for GraphDatabaseManager initialization."""

    def test_should_create_manager_with_config(self) -> None:
        """Should create manager with configuration from AppConfig."""
        with patch("src.database.graph_db.GraphDatabase") as mock_driver:
            mock_driver.driver.return_value = MagicMock()
            
            manager = GraphDatabaseManager()
            
            assert manager is not None
            assert manager._driver is not None

    def test_should_use_custom_uri(self) -> None:
        """Should use custom URI when provided."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_db.driver.return_value = MagicMock()
            
            manager = GraphDatabaseManager(
                uri="bolt://custom:7687",
                username="user",
                password="pass"
            )
            
            mock_db.driver.assert_called_once_with(
                "bolt://custom:7687",
                auth=("user", "pass")
            )


class TestGraphDatabaseManagerConnection:
    """Tests for connection management."""

    def test_should_verify_connectivity_when_healthy(self) -> None:
        """Should return True when database is reachable."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity.return_value = None
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            assert manager.is_healthy() is True

    def test_should_return_false_when_connection_fails(self) -> None:
        """Should return False when database is unreachable."""
        from neo4j.exceptions import ServiceUnavailable
        
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Connection failed")
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            assert manager.is_healthy() is False

    def test_should_close_driver_on_context_manager_exit(self) -> None:
        """Should properly close driver when used as context manager."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            
            with GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            ) as manager:
                pass
            
            mock_driver.close.assert_called_once()


class TestGraphDatabaseManagerQueries:
    """Tests for query execution."""

    def test_should_execute_cypher_query(self) -> None:
        """Should execute a Cypher query and return results."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value = mock_result
            mock_result.data.return_value = [{"name": "Test"}]
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            result = manager.execute_query("MATCH (n) RETURN n.name AS name")
            
            assert result == [{"name": "Test"}]

    def test_should_execute_query_with_parameters(self) -> None:
        """Should pass parameters to Cypher query."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value = mock_result
            mock_result.data.return_value = [{"name": "Singapore"}]
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            result = manager.execute_query(
                "MATCH (n:Location {name: $name}) RETURN n",
                {"name": "Singapore"}
            )
            
            mock_session.run.assert_called_once_with(
                "MATCH (n:Location {name: $name}) RETURN n",
                {"name": "Singapore"}
            )


class TestGraphDatabaseManagerMerge:
    """Tests for MERGE operations (idempotency)."""

    def test_should_merge_node_without_duplicates(self) -> None:
        """Should use MERGE to create nodes idempotently."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value = mock_result
            mock_result.single.return_value = {"n": MagicMock()}
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            result = manager.merge_node(
                label="Company",
                properties={"name": "TechFlow Inc."}
            )
            
            # Verify MERGE was called, not CREATE
            call_args = mock_session.run.call_args[0][0]
            assert "MERGE" in call_args
            assert "Company" in call_args

    def test_should_merge_relationship(self) -> None:
        """Should create relationship using MERGE for idempotency."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value = mock_result
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            manager.merge_relationship(
                from_label="Company",
                from_props={"name": "TechFlow Inc."},
                to_label="Product",
                to_props={"name": "FlowChips"},
                rel_type="MANUFACTURES"
            )
            
            call_args = mock_session.run.call_args[0][0]
            assert "MERGE" in call_args
            assert "MANUFACTURES" in call_args


class TestGraphDatabaseManagerGraphTraversal:
    """Tests for graph traversal operations."""

    def test_should_find_n_hop_neighbors(self) -> None:
        """Should find neighbors within N hops."""
        with patch("src.database.graph_db.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value = mock_result
            mock_result.data.return_value = [
                {"node": {"name": "FlowChips"}, "path_length": 1},
                {"node": {"name": "GlobalTech"}, "path_length": 2}
            ]
            
            manager = GraphDatabaseManager(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            
            result = manager.find_neighbors(
                label="Location",
                properties={"name": "Singapore"},
                max_hops=2
            )
            
            assert len(result) == 2
            # Verify query uses variable-length path pattern
            call_args = mock_session.run.call_args[0][0]
            assert "*1..2" in call_args or "*..2" in call_args
