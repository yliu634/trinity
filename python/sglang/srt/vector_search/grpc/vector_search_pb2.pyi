from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Stage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STAGE_UNSPECIFIED: _ClassVar[Stage]
    PREFILL: _ClassVar[Stage]
    DECODE: _ClassVar[Stage]
STAGE_UNSPECIFIED: Stage
PREFILL: Stage
DECODE: Stage

class SearchRequest(_message.Message):
    __slots__ = ("request_id", "stage", "text", "topk", "deadline_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOPK_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    stage: Stage
    text: str
    topk: int
    deadline_ms: int
    def __init__(self, request_id: _Optional[str] = ..., stage: _Optional[_Union[Stage, str]] = ..., text: _Optional[str] = ..., topk: _Optional[int] = ..., deadline_ms: _Optional[int] = ...) -> None: ...

class RetrievedDoc(_message.Message):
    __slots__ = ("doc_id", "text", "score")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    text: str
    score: float
    def __init__(self, doc_id: _Optional[str] = ..., text: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("request_id", "docs", "error_message")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DOCS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    docs: _containers.RepeatedCompositeFieldContainer[RetrievedDoc]
    error_message: str
    def __init__(self, request_id: _Optional[str] = ..., docs: _Optional[_Iterable[_Union[RetrievedDoc, _Mapping]]] = ..., error_message: _Optional[str] = ...) -> None: ...
