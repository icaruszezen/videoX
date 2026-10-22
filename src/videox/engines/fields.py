from __future__ import annotations

from videox.errors import VideoXError

VIDEO = ["video"]
IMAGE = ["image"]

Condition = dict[str, list[object]]


def string_field(
    title: str,
    default: str,
    *,
    group: str,
    description: str,
    choices: list[tuple[str, str]] | None = None,
    media: list[str] = VIDEO,
) -> dict:
    field: dict = {
        "type": "string",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
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
    step: int | None = None,
    media: list[str] = VIDEO,
) -> dict:
    field: dict = {
        "type": "integer",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
        "minimum": minimum,
        "description": description,
    }
    if maximum is not None:
        field["maximum"] = maximum
    if step is not None:
        # Valid values are minimum + k * step; the UI uses it for the spinner and checks.
        field["step"] = step
    return field


def number_field(
    title: str,
    default: float,
    *,
    group: str,
    description: str,
    minimum: float,
    maximum: float,
    media: list[str] = VIDEO,
) -> dict:
    return {
        "type": "number",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
        "minimum": minimum,
        "maximum": maximum,
        "description": description,
    }


def boolean_field(
    title: str,
    default: bool,
    *,
    group: str,
    description: str,
    media: list[str] = VIDEO,
) -> dict:
    return {
        "type": "boolean",
        "title": title,
        "default": default,
        "group": group,
        "media": media,
        "description": description,
    }


def conditional(
    field: dict,
    *,
    visible_when: Condition | None = None,
    overrides: list[tuple[Condition, dict]] | None = None,
) -> dict:
    """Tie a field to the value of an earlier field, usually the model choice.

    ``visible_when`` maps earlier fields to the values that show this one. A
    hidden field is not sent and the engine does not read it.
    Each override is ``(condition, changes)``; the first condition that holds
    replaces keys such as ``default``, ``maximum``, ``enum`` or ``description``.
    The desktop form evaluates the same rules, in property order.
    """
    updated = dict(field)
    if visible_when:
        updated["visible_when"] = {key: list(values) for key, values in visible_when.items()}
    if overrides:
        updated["overrides"] = [
            {"when": {key: list(values) for key, values in condition.items()}, **changes}
            for condition, changes in overrides
        ]
    return updated


def choice_notes(field: dict, notes: dict[str, str]) -> dict:
    """Attach a one-line note per choice. The form shows the note under the select."""
    return {**field, "enum_descriptions": dict(notes)}


def schema(groups: list[tuple[str, str]], properties: dict) -> dict:
    return {
        "type": "object",
        "groups": [{"id": group_id, "title": title} for group_id, title in groups],
        "properties": properties,
    }


def condition_holds(condition: Condition, values: dict[str, object]) -> bool:
    return all(_matches(values.get(key), allowed) for key, allowed in condition.items())


def resolve_field(field: dict, values: dict[str, object]) -> dict | None:
    """Apply ``visible_when`` and ``overrides`` against the values resolved so far."""
    visible_when = field.get("visible_when")
    if visible_when and not condition_holds(visible_when, values):
        return None
    for override in field.get("overrides") or []:
        if condition_holds(override.get("when") or {}, values):
            changes = {key: value for key, value in override.items() if key != "when"}
            return {**field, **changes}
    return field


def resolve_params(params_schema: dict, params: dict, media: str) -> dict[str, object]:
    """Return every field that applies to this job, validated, with defaults filled in.

    Fields hidden by ``visible_when`` or meant for the other media are left out,
    even if the caller sent them.
    """
    values: dict[str, object] = {}
    for key, field in (params_schema.get("properties") or {}).items():
        if media not in (field.get("media") or [media]):
            continue
        resolved = resolve_field(field, values)
        if resolved is None:
            continue
        if key in params:
            values[key] = _validate(params[key], resolved)
        else:
            values[key] = resolved.get("default")
    return values


def _matches(value: object, allowed: list[object]) -> bool:
    for item in allowed:
        # bool is an int in Python; keep True from matching 1.
        if isinstance(item, bool) or isinstance(value, bool):
            if value is item:
                return True
        elif value == item:
            return True
    return False


def _validate(value: object, field: dict) -> object:
    label = field.get("title") or "参数"
    kind = field.get("type")
    if kind == "boolean":
        if not isinstance(value, bool):
            raise VideoXError(f"{label} 需要是布尔值")
        return value
    if kind in {"integer", "number"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VideoXError(f"{label} 需要是{'整数' if kind == 'integer' else '数字'}")
        if kind == "integer":
            if isinstance(value, float) and not value.is_integer():
                raise VideoXError(f"{label} 需要是整数")
            value = int(value)
        else:
            value = float(value)
        low = field.get("minimum")
        high = field.get("maximum")
        if low is not None and value < low:
            raise VideoXError(f"{label} 不能小于 {low:g}")
        if high is not None and value > high:
            raise VideoXError(f"{label} 不能大于 {high:g}")
        return value
    if not isinstance(value, str):
        raise VideoXError(f"{label} 需要是字符串")
    choices = field.get("enum")
    if choices is not None and value not in choices:
        raise VideoXError(f"{label} 无法识别: {value}")
    return value
