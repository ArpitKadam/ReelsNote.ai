from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class FinalState(TypedDict, total=False):
    url: str
    title: str
    caption: str
    video_path: Optional[str]
    has_audio: bool
    transcript: Optional[str]
    video_explanation: Optional[str]
    report: Optional[str]
