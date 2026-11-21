"""JSON schemas (draft-07 subset) for actions, submissions and public records, plus a
small dependency-free validator for that subset."""
from __future__ import annotations

import re
from typing import Any

from .. import FINDING_TYPES

LIMITS = {
    "max_steps": 100,
    "page_size_max": 50,
    "max_findings": 30,
    "max_evidence_ids": 10,
    "max_affected_ids": 20,
    "max_explanation_chars": 1000,
    "max_id_chars": 64,
    "max_action_bytes": 65536,
    "time_limit_seconds": 60,
    "amount_tolerance_minor": 1,
}

TABLES = ("ledger", "invoices", "payments", "fx")

_ID = {"type": "string", "minLength": 1, "maxLength": LIMITS["max_id_chars"]}
_NULLABLE_ID = {"type": ["string", "null"], "maxLength": LIMITS["max_id_chars"]}
_CUR = {"type": "string", "pattern": "^[A-Z]{3}$"}
_INT = {"type": "integer"}

FINDING_SCHEMA = {
    "type": "object",
    "required": ["type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"],
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "enum": list(FINDING_TYPES)},
        "affected_ids": {"type": "array", "minItems": 1, "maxItems": LIMITS["max_affected_ids"], "items": _ID},
        "claimed_amount_minor": {"type": "integer", "minimum": 0, "maximum": 10**15},
        "currency": _CUR,
        "evidence_ids": {"type": "array", "maxItems": LIMITS["max_evidence_ids"], "items": _ID},
        "explanation": {"type": "string", "maxLength": LIMITS["max_explanation_chars"]},
    },
}

SUBMIT_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action", "episode_id", "findings"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["submit"]},
        "episode_id": _ID,
        "findings": {"type": "array", "maxItems": LIMITS["max_findings"], "items": FINDING_SCHEMA},
    },
}

ACTION_SCHEMAS = {
    "list_records": {
        "type": "object", "required": ["action", "table"], "additionalProperties": False,
        "properties": {
            "action": {"type": "string"},
            "table": {"type": "string", "enum": list(TABLES)},
            "cursor": {"type": ["string", "null"], "maxLength": 32},
            "limit": {"type": "integer", "minimum": 1},
        },
    },
    "get_record": {
        "type": "object", "required": ["action", "table", "id"], "additionalProperties": False,
        "properties": {
            "action": {"type": "string"},
            "table": {"type": "string", "enum": list(TABLES)},
            "id": _ID,
        },
    },
    "get_document": {
        "type": "object", "required": ["action", "doc_id"], "additionalProperties": False,
        "properties": {"action": {"type": "string"}, "doc_id": _ID},
    },
    "calculator": {
        "type": "object", "required": ["action", "expression"], "additionalProperties": False,
        "properties": {"action": {"type": "string"}, "expression": {"type": "string", "maxLength": 256}},
    },
    "submit": SUBMIT_ACTION_SCHEMA,
}

LEDGER_ROW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                 "entity_id", "invoice_id", "payment_id"],
    "properties": {
        "row_id": _ID, "timestamp": {"type": "string"}, "account": {"type": "string", "pattern": "^[0-9]{4}$"},
        "debit_minor": {"type": "integer", "minimum": 0}, "credit_minor": {"type": "integer", "minimum": 0},
        "currency": _CUR, "entity_id": _NULLABLE_ID, "invoice_id": _NULLABLE_ID, "payment_id": _NULLABLE_ID,
    },
}
LINE_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["sku", "description", "quantity", "unit_price_minor", "amount_minor"],
    "properties": {"sku": _ID, "description": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1},
                   "unit_price_minor": _INT, "amount_minor": _INT},
}
INVOICE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor", "line_items"],
    "properties": {"doc_id": _ID, "invoice_id": _ID, "entity_id": _ID, "issued_at": {"type": "string"},
                   "due_at": {"type": "string"}, "currency": _CUR, "total_minor": _INT,
                   "line_items": {"type": "array", "minItems": 1, "items": LINE_ITEM_SCHEMA}},
}
PAYMENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["doc_id", "payment_id", "entity_id", "timestamp", "currency", "amount_minor", "invoice_refs"],
    "properties": {"doc_id": _ID, "payment_id": _ID, "entity_id": _ID, "timestamp": {"type": "string"},
                   "currency": _CUR, "amount_minor": _INT,
                   "invoice_refs": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["invoice_id", "amount_minor"],
                       "properties": {"invoice_id": _ID, "amount_minor": _INT}}}},
}
FX_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["date", "source_currency", "target_currency", "rate"],
    "properties": {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}, "source_currency": _CUR,
                   "target_currency": _CUR, "rate": {"type": "string", "pattern": r"^\d+\.\d+$"}},
}

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable violations (empty if valid)."""
    errs: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES[x](value) for x in types):
            return [f"{path}: expected {'/'.join(types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: pattern mismatch")
    if _TYPES["integer"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: above maximum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: too many items")
        if "items" in schema:
            for i, item in enumerate(value):
                errs.extend(validate(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs.extend(validate(v, props[k], f"{path}.{k}"))
            elif schema.get("additionalProperties", True) is False:
                errs.append(f"{path}: unexpected key")
    return errs


__all__ = ["LIMITS", "TABLES", "ACTION_SCHEMAS", "SUBMIT_ACTION_SCHEMA", "FINDING_SCHEMA",
           "LEDGER_ROW_SCHEMA", "INVOICE_SCHEMA", "PAYMENT_SCHEMA", "FX_SCHEMA", "validate"]
