"""Retry pacing for hosted reference generation.

Worth pinning because the failure is silent: `generate_reference` raising is caught by
`object_references.generate_for_scan`, which writes a PLACEHOLDER reference and reports the run as
finished. A rate limit that outruns the backoff therefore degrades the room without failing it —
which is exactly what a higher LR_IMAGE_WORKERS makes more likely.
"""

from __future__ import annotations

import pytest

from lrauthor.pipeline.measure.ingest.references import image_gen


class _Response:
    def __init__(self, headers):
        self.headers = headers


class RateLimited(Exception):
    status_code = 429

    def __init__(self, headers=None):
        super().__init__("rate limit exceeded")
        self.response = _Response(headers or {})


def test_default_pool_covers_a_normal_room_in_one_wave():
    """17 objects used to take three waves at 6 workers."""
    assert image_gen.MAX_IMAGE_WORKERS >= 17
    assert image_gen.image_workers(17) == 17


def test_pool_never_exceeds_the_batch():
    assert image_gen.image_workers(3) == 3
    assert image_gen.image_workers(0) == 1


def test_rate_limit_waits_far_longer_than_a_transient_error():
    """The old ladder spent every attempt inside nine seconds, well short of a per-minute budget."""
    transient = image_gen.retry_delay(RuntimeError("boom"), 0)
    limited = image_gen.retry_delay(RateLimited(), 0)
    assert transient == pytest.approx(1.5)
    assert limited >= 20.0


def test_rate_limit_backoff_grows():
    delays = [image_gen.retry_delay(RateLimited(), i) for i in range(3)]
    assert delays == sorted(delays)
    assert delays[0] < delays[-1]


def test_backoff_is_capped():
    assert image_gen.retry_delay(RateLimited(), 12) <= image_gen._MAX_BACKOFF


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"retry-after": "30"}, 30.5),
        ({"retry-after-ms": "45000"}, 45.5),
        ({"retry-after": "not-a-number"}, 20.0),
    ],
)
def test_server_retry_after_is_honoured(headers, expected):
    """Trust the server's own number over our guess; fall back to the ladder if it is unusable."""
    assert image_gen.retry_delay(RateLimited(headers), 0) == pytest.approx(expected)


def test_retry_after_is_capped_too():
    assert image_gen.retry_delay(RateLimited({"retry-after": "9999"}), 0) == image_gen._MAX_BACKOFF


def test_no_sleep_after_the_final_attempt(tmp_path, monkeypatch):
    """The loop used to sleep once more after the last try — pure dead time per failed object."""
    slept: list[float] = []
    monkeypatch.setattr(image_gen.time, "sleep", slept.append)

    class _Client:
        def __init__(self, *_, **__):
            self.images = self

        def edit(self, **__):
            raise RuntimeError("boom")

    monkeypatch.setattr("openai.OpenAI", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"x")

    with pytest.raises(RuntimeError):
        image_gen.generate_reference(sheet, "prompt", tmp_path / "out.png", retries=3)

    assert len(slept) == 2  # three attempts, two waits between them


class OutOfCredit(Exception):
    """The 429 OpenAI sends for an empty balance — same status as a rate limit, opposite remedy."""

    status_code = 429

    def __init__(self):
        super().__init__(
            "Error code: 429 - {'error': {'message': 'You have no credits remaining.', "
            "'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}}"
        )
        self.body = {
            "error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}
        }
        self.response = _Response({})


def test_empty_balance_is_told_apart_from_a_rate_limit():
    assert image_gen.is_out_of_credit(OutOfCredit()) is True
    assert image_gen.is_out_of_credit(RateLimited()) is False
    assert image_gen.is_out_of_credit(RuntimeError("boom")) is False


def test_empty_balance_fails_on_the_first_attempt(tmp_path, monkeypatch):
    """Waiting 20s then 40s for a balance that will never refill is a minute wasted per object."""
    slept: list[float] = []
    monkeypatch.setattr(image_gen.time, "sleep", slept.append)
    calls: list[int] = []

    class _Client:
        def __init__(self, *_, **__):
            self.images = self

        def edit(self, **__):
            calls.append(1)
            raise OutOfCredit()

    monkeypatch.setattr("openai.OpenAI", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"x")

    with pytest.raises(RuntimeError, match="no credits remaining"):
        image_gen.generate_reference(sheet, "prompt", tmp_path / "out.png", retries=3)

    assert calls == [1]  # one attempt, not three
    assert slept == []   # and no backoff at all
