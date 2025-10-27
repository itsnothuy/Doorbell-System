#!/usr/bin/env python3
"""
Query Builder - SQL Query Construction Utilities

Provides utilities for building SQL queries safely with parameter binding,
filtering, sorting, and pagination support.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class Operator(Enum):
    """Query operators."""
    EQ = "="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    LIKE = "LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"


class SortOrder(Enum):
    """Sort order."""
    ASC = "ASC"
    DESC = "DESC"


@dataclass
class QueryFilter:
    """Represents a query filter condition."""
    field: str
    operator: Operator
    value: Any = None
    
    def to_sql(self) -> Tuple[str, List[Any]]:
        """
        Convert filter to SQL WHERE clause.
        
        Returns:
            Tuple of (sql_clause, parameters)
        """
        if self.operator in (Operator.IS_NULL, Operator.IS_NOT_NULL):
            return f"{self.field} {self.operator.value}", []
        
        elif self.operator == Operator.IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError("IN operator requires list/tuple value")
            placeholders = ','.join(['?' for _ in self.value])
            return f"{self.field} IN ({placeholders})", list(self.value)
        
        elif self.operator == Operator.NOT_IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError("NOT IN operator requires list/tuple value")
            placeholders = ','.join(['?' for _ in self.value])
            return f"{self.field} NOT IN ({placeholders})", list(self.value)
        
        elif self.operator == Operator.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                raise ValueError("BETWEEN operator requires two-element list/tuple")
            return f"{self.field} BETWEEN ? AND ?", list(self.value)
        
        else:
            return f"{self.field} {self.operator.value} ?", [self.value]


@dataclass
class QuerySort:
    """Represents query sorting."""
    field: str
    order: SortOrder = SortOrder.ASC
    
    def to_sql(self) -> str:
        """Convert to SQL ORDER BY clause."""
        return f"{self.field} {self.order.value}"


@dataclass
class QueryBuilder:
    """
    SQL query builder for constructing safe, parameterized queries.
    
    Supports filtering, sorting, pagination, and joins.
    """
    table: str
    filters: List[QueryFilter] = field(default_factory=list)
    sorts: List[QuerySort] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    select_fields: List[str] = field(default_factory=list)
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """
        Set fields to select.
        
        Args:
            fields: Field names to select
            
        Returns:
            Self for chaining
        """
        self.select_fields.extend(fields)
        return self
    
    def filter(self, field: str, operator: Operator, value: Any = None) -> 'QueryBuilder':
        """
        Add a filter condition.
        
        Args:
            field: Field name
            operator: Comparison operator
            value: Value to compare (not needed for IS NULL/IS NOT NULL)
            
        Returns:
            Self for chaining
        """
        self.filters.append(QueryFilter(field, operator, value))
        return self
    
    def filter_eq(self, field: str, value: Any) -> 'QueryBuilder':
        """Add equals filter."""
        return self.filter(field, Operator.EQ, value)
    
    def filter_ne(self, field: str, value: Any) -> 'QueryBuilder':
        """Add not-equals filter."""
        return self.filter(field, Operator.NE, value)
    
    def filter_gt(self, field: str, value: Any) -> 'QueryBuilder':
        """Add greater-than filter."""
        return self.filter(field, Operator.GT, value)
    
    def filter_gte(self, field: str, value: Any) -> 'QueryBuilder':
        """Add greater-than-or-equal filter."""
        return self.filter(field, Operator.GTE, value)
    
    def filter_lt(self, field: str, value: Any) -> 'QueryBuilder':
        """Add less-than filter."""
        return self.filter(field, Operator.LT, value)
    
    def filter_lte(self, field: str, value: Any) -> 'QueryBuilder':
        """Add less-than-or-equal filter."""
        return self.filter(field, Operator.LTE, value)
    
    def filter_like(self, field: str, pattern: str) -> 'QueryBuilder':
        """Add LIKE filter."""
        return self.filter(field, Operator.LIKE, pattern)
    
    def filter_in(self, field: str, values: List[Any]) -> 'QueryBuilder':
        """Add IN filter."""
        return self.filter(field, Operator.IN, values)
    
    def filter_not_in(self, field: str, values: List[Any]) -> 'QueryBuilder':
        """Add NOT IN filter."""
        return self.filter(field, Operator.NOT_IN, values)
    
    def filter_is_null(self, field: str) -> 'QueryBuilder':
        """Add IS NULL filter."""
        return self.filter(field, Operator.IS_NULL)
    
    def filter_is_not_null(self, field: str) -> 'QueryBuilder':
        """Add IS NOT NULL filter."""
        return self.filter(field, Operator.IS_NOT_NULL)
    
    def filter_between(self, field: str, start: Any, end: Any) -> 'QueryBuilder':
        """Add BETWEEN filter."""
        return self.filter(field, Operator.BETWEEN, [start, end])
    
    def order_by(self, field: str, order: SortOrder = SortOrder.ASC) -> 'QueryBuilder':
        """
        Add sorting.
        
        Args:
            field: Field to sort by
            order: Sort order (ASC or DESC)
            
        Returns:
            Self for chaining
        """
        self.sorts.append(QuerySort(field, order))
        return self
    
    def order_by_asc(self, field: str) -> 'QueryBuilder':
        """Add ascending sort."""
        return self.order_by(field, SortOrder.ASC)
    
    def order_by_desc(self, field: str) -> 'QueryBuilder':
        """Add descending sort."""
        return self.order_by(field, SortOrder.DESC)
    
    def paginate(self, limit: int, offset: int = 0) -> 'QueryBuilder':
        """
        Add pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Self for chaining
        """
        self.limit = limit
        self.offset = offset
        return self
    
    def build_select(self) -> Tuple[str, List[Any]]:
        """
        Build SELECT query.
        
        Returns:
            Tuple of (sql_query, parameters)
        """
        # SELECT clause
        if self.select_fields:
            select_clause = f"SELECT {', '.join(self.select_fields)}"
        else:
            select_clause = "SELECT *"
        
        # FROM clause
        from_clause = f"FROM {self.table}"
        
        # WHERE clause
        where_clause = ""
        params: List[Any] = []
        
        if self.filters:
            conditions = []
            for filter_obj in self.filters:
                condition, filter_params = filter_obj.to_sql()
                conditions.append(condition)
                params.extend(filter_params)
            
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # ORDER BY clause
        order_clause = ""
        if self.sorts:
            order_parts = [sort.to_sql() for sort in self.sorts]
            order_clause = "ORDER BY " + ", ".join(order_parts)
        
        # LIMIT/OFFSET clause
        limit_clause = ""
        if self.limit is not None:
            limit_clause = f"LIMIT {self.limit}"
            if self.offset:
                limit_clause += f" OFFSET {self.offset}"
        
        # Combine all parts
        query_parts = [select_clause, from_clause]
        if where_clause:
            query_parts.append(where_clause)
        if order_clause:
            query_parts.append(order_clause)
        if limit_clause:
            query_parts.append(limit_clause)
        
        query = " ".join(query_parts)
        
        return query, params
    
    def build_count(self) -> Tuple[str, List[Any]]:
        """
        Build COUNT query.
        
        Returns:
            Tuple of (sql_query, parameters)
        """
        # WHERE clause
        where_clause = ""
        params: List[Any] = []
        
        if self.filters:
            conditions = []
            for filter_obj in self.filters:
                condition, filter_params = filter_obj.to_sql()
                conditions.append(condition)
                params.extend(filter_params)
            
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # Build query
        query = f"SELECT COUNT(*) FROM {self.table}"
        if where_clause:
            query += f" {where_clause}"
        
        return query, params
    
    def build_delete(self) -> Tuple[str, List[Any]]:
        """
        Build DELETE query.
        
        Returns:
            Tuple of (sql_query, parameters)
        """
        if not self.filters:
            raise ValueError("DELETE query requires at least one filter for safety")
        
        # WHERE clause
        conditions = []
        params: List[Any] = []
        
        for filter_obj in self.filters:
            condition, filter_params = filter_obj.to_sql()
            conditions.append(condition)
            params.extend(filter_params)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"DELETE FROM {self.table} {where_clause}"
        
        return query, params
    
    def build_update(self, updates: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build UPDATE query.
        
        Args:
            updates: Dictionary of field: value pairs to update
            
        Returns:
            Tuple of (sql_query, parameters)
        """
        if not updates:
            raise ValueError("UPDATE query requires update values")
        
        if not self.filters:
            raise ValueError("UPDATE query requires at least one filter for safety")
        
        # SET clause
        set_parts = []
        params: List[Any] = []
        
        for field, value in updates.items():
            set_parts.append(f"{field} = ?")
            params.append(value)
        
        set_clause = "SET " + ", ".join(set_parts)
        
        # WHERE clause
        conditions = []
        for filter_obj in self.filters:
            condition, filter_params = filter_obj.to_sql()
            conditions.append(condition)
            params.extend(filter_params)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"UPDATE {self.table} {set_clause} {where_clause}"
        
        return query, params
    
    def reset(self) -> 'QueryBuilder':
        """Reset query builder to initial state."""
        self.filters.clear()
        self.sorts.clear()
        self.select_fields.clear()
        self.limit = None
        self.offset = None
        return self


def build_insert_query(table: str, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build INSERT query.
    
    Args:
        table: Table name
        data: Dictionary of field: value pairs
        
    Returns:
        Tuple of (sql_query, parameters)
    """
    if not data:
        raise ValueError("INSERT query requires data")
    
    fields = list(data.keys())
    values = list(data.values())
    placeholders = ','.join(['?' for _ in fields])
    
    query = f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({placeholders})"
    
    return query, values


def build_bulk_insert_query(table: str, rows: List[Dict[str, Any]]) -> Tuple[str, List[List[Any]]]:
    """
    Build bulk INSERT query.
    
    Args:
        table: Table name
        rows: List of dictionaries with field: value pairs
        
    Returns:
        Tuple of (sql_query, list_of_parameter_lists)
    """
    if not rows:
        raise ValueError("Bulk INSERT requires at least one row")
    
    # Use first row to get field names
    fields = list(rows[0].keys())
    placeholders = ','.join(['?' for _ in fields])
    
    query = f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({placeholders})"
    
    # Extract values for each row
    params_list = []
    for row in rows:
        values = [row.get(field) for field in fields]
        params_list.append(values)
    
    return query, params_list
