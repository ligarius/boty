import json
import logging

from bot.obs.logging import JsonFormatter


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord(
        name="test-logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processed %s items",
        args=(5,),
        exc_info=None,
    )
    record.user_id = 123
    record.context = {"state": "ok"}

    formatter = JsonFormatter()

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "Processed 5 items"
    assert payload["logger"] == "test-logger"
    assert payload["user_id"] == 123
    assert payload["context"] == {"state": "ok"}
