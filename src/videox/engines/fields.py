from __future__ import annotations

VIDEO = ["video"]


def string_field(
    title: str,
    default: str,
    *,
    group: str,
    description: str,
    choices: list[tuple[str, str]] | None = None,
) -> dict:
    field: dict = {
        "type": "string",
        "title": title,
        "default": default,
        "group": group,
        "media": VIDEO,
        "description": description,
    }
    if choices is not None:
        field["enum"] = [value for value, _label in choices]
        field["enum_labels"] = {value: label for value, label in choices}
    return field


def integer_field(
    title: str,
    default: int,
    *,
    group: str,
    description: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> dict:
    field: dict = {
        "type": "integer",
        "title": title,
        "default": default,
        "group": group,
        "media": VIDEO,
        "minimum": minimum,
        "description": description,
    }
    if maximum is not None:
        field["maximum"] = maximum
    return field


def number_field(
    title: str,
    default: float,
    *,
    group: str,
    description: str,
    minimum: float,
    maximum: float,
) -> dict:
    return {
        "type": "number",
        "title": title,
        "default": default,
        "group": group,
        "media": VIDEO,
        "minimum": minimum,
        "maximum": maximum,
        "description": description,
    }


def boolean_field(title: str, default: bool, *, group: str, description: str) -> dict:
    return {
        "type": "boolean",
        "title": title,
        "default": default,
        "group": group,
        "media": VIDEO,
        "description": description,
    }


def schema(groups: list[tuple[str, str]], properties: dict) -> dict:
    return {
        "type": "object",
        "groups": [{"id": group_id, "title": title} for group_id, title in groups],
        "properties": properties,
    }
