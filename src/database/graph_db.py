"""
Neo4j Graph Database Manager for Atlas-GRAG.

Handles all interactions with the Neo4j knowledge graph including
connection management, query execution, and MERGE operations for idempotency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import get_config


class GraphDatabaseError(Exception):
    """Custom exception for graph database errors."""
    pass


class GraphDatabaseManager:
    """
    Manager for Neo4j graph database operations.
    
    Provides connection management, query execution, and idempotent
    MERGE operations for the supply chain knowledge graph.
    
    Usage:
        with GraphDatabaseManager() as manager:
            results = manager.execute_query("MATCH (n) RETURN n LIMIT 10")
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Initialize the graph database manager.
        
        Args:
            uri: Neo4j bolt URI (defaults to config)
            username: Neo4j username (defaults to config)
            password: Neo4j password (defaults to config)
            database: Neo4j database name (defaults to config)
        """
        config = get_config().neo4j
        
        self._uri = uri or config.uri
        self._username = username or config.username
        self._password = password or config.password
        self._database = database or config.database
        
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password)
        )
    
    def __enter__(self) -> GraphDatabaseManager:
        """Context manager entry."""
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Context manager exit - close the driver."""
        self.close()
    
    def close(self) -> None:
        """Close the database driver."""
        if self._driver:
            self._driver.close()
    
    def is_healthy(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            True if database is reachable, False otherwise.
        """
        try:
            self._driver.verify_connectivity()
            return True
        except ServiceUnavailable:
            return False
        except Exception:
            return False
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        parameters = parameters or {}
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters)
            return result.data()
    
    def merge_node(
        self,
        label: str,
        properties: Dict[str, Any],
        on_create: Optional[Dict[str, Any]] = None,
        on_match: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge a node using MERGE for idempotency.
        
        Args:
            label: Node label (e.g., 'Company', 'Product')
            properties: Properties to match on
            on_create: Properties to set only on creation
            on_match: Properties to set only on match
            
        Returns:
            The merged node dictionary
        """
        # Build the MERGE query
        prop_string = ", ".join(f"{k}: ${k}" for k in properties.keys())
        query = f"MERGE (n:{label} {{{prop_string}}})"
        
        params: Dict[str, Any] = dict(properties)
        
        if on_create:
            create_string = ", ".join(f"n.{k} = $create_{k}" for k in on_create.keys())
            query += f" ON CREATE SET {create_string}"
            params.update({f"create_{k}": v for k, v in on_create.items()})
        
        if on_match:
            match_string = ", ".join(f"n.{k} = $match_{k}" for k in on_match.keys())
            query += f" ON MATCH SET {match_string}"
            params.update({f"match_{k}": v for k, v in on_match.items()})
        
        query += " RETURN n"
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return result.single()
    
    def merge_relationship(
        self,
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        rel_type: str,
        rel_props: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Merge a relationship between two nodes using MERGE for idempotency.
        
        Args:
            from_label: Source node label
            from_props: Source node matching properties
            to_label: Target node label
            to_props: Target node matching properties
            rel_type: Relationship type (e.g., 'MANUFACTURES')
            rel_props: Optional relationship properties
        """
        # Build property match strings
        from_prop_string = ", ".join(f"{k}: $from_{k}" for k in from_props.keys())
        to_prop_string = ", ".join(f"{k}: $to_{k}" for k in to_props.keys())
        
        query = f"""
        MERGE (a:{from_label} {{{from_prop_string}}})
        MERGE (b:{to_label} {{{to_prop_string}}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        
        params: Dict[str, Any] = {}
        params.update({f"from_{k}": v for k, v in from_props.items()})
        params.update({f"to_{k}": v for k, v in to_props.items()})
        
        if rel_props:
            set_string = ", ".join(f"r.{k} = $rel_{k}" for k in rel_props.keys())
            query += f" SET {set_string}"
            params.update({f"rel_{k}": v for k, v in rel_props.items()})
        
        query += " RETURN r"
        
        with self._driver.session(database=self._database) as session:
            session.run(query, params)
    
    def find_neighbors(
        self,
        label: str,
        properties: Dict[str, Any],
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find all nodes within N hops of a starting node.
        
        Args:
            label: Starting node label
            properties: Starting node properties to match
            max_hops: Maximum number of relationship hops (default: 2)
            
        Returns:
            List of neighbor nodes with path length
        """
        prop_string = ", ".join(f"{k}: ${k}" for k in properties.keys())
        
        query = f"""
        MATCH (start:{label} {{{prop_string}}})
        MATCH path = (start)-[*1..{max_hops}]-(neighbor)
        WHERE neighbor <> start
        RETURN DISTINCT neighbor AS node, length(path) AS path_length
        ORDER BY path_length
        """
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, dict(properties))
            return result.data()
    
    def get_paths_between(
        self,
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all paths between two nodes.
        
        Args:
            from_label: Source node label
            from_props: Source node properties
            to_label: Target node label
            to_props: Target node properties
            max_hops: Maximum path length
            
        Returns:
            List of paths with nodes and relationships
        """
        from_prop_string = ", ".join(f"{k}: $from_{k}" for k in from_props.keys())
        to_prop_string = ", ".join(f"{k}: $to_{k}" for k in to_props.keys())
        
        query = f"""
        MATCH path = (a:{from_label} {{{from_prop_string}}})-[*1..{max_hops}]-(b:{to_label} {{{to_prop_string}}})
        RETURN [n in nodes(path) | n] AS nodes,
               [r in relationships(path) | type(r)] AS relationships,
               length(path) AS path_length
        ORDER BY path_length
        LIMIT 10
        """
        
        params: Dict[str, Any] = {}
        params.update({f"from_{k}": v for k, v in from_props.items()})
        params.update({f"to_{k}": v for k, v in to_props.items()})
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return result.data()
