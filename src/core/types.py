from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

class AppConfig(BaseModel):
    """全局应用配置模型，定义了 config.json 中的所有字段及其默认值。"""
    port_api: int = 2156
    max_retries: int = 2
    error_dir: str = "errors"
    debug: bool = False
    log_dir: str = "logs"
    anti_tracking: bool = False
    drop_max_tokens: bool = True
    log_payload: bool = False  # 为 true 时将原始 fetch/response 保存到 logs 目录
    api_key: str | None = None

    model_config = ConfigDict(extra="allow")

class APIKeyInfo(BaseModel):
    name: str
    is_active: bool


class JSONSchemaProperty(BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None
    required: list[str] | None = None
    
    model_config = ConfigDict(extra="allow")

class JSONSchema(BaseModel):
    type: str
    properties: dict[str, JSONSchemaProperty] | None = None
    required: list[str] | None = None
    description: str | None = None
    
    model_config = ConfigDict(extra="allow")

class GenerationConfig(BaseModel):
    maxOutputTokens: int | None = Field(None, alias="maxOutputTokens")
    stopSequences: list[str] | None = Field(None, alias="stopSequences")
    topP: float | None = Field(None, alias="topP")
    topK: int | None = Field(None, alias="topK")
    responseMimeType: str | None = Field(None, alias="responseMimeType")
    responseSchema: JSONSchema | None = Field(None, alias="responseSchema")
    candidateCount: int | None = Field(None, alias="candidateCount")
    presencePenalty: float | None = Field(None, alias="presencePenalty")
    frequencyPenalty: float | None = Field(None, alias="frequencyPenalty")
    responseLogprobs: bool | None = Field(None, alias="responseLogprobs")
    speechConfig: dict[str, str | int | float] | None = Field(None, alias="speechConfig")
    audioTimestamp: bool | None = Field(None, alias="audioTimestamp")
    enableEnhancedCivicAnswers: bool | None = Field(None, alias="enableEnhancedCivicAnswers")

    model_config = ConfigDict(populate_by_name=True, extra="allow")

class SafetySetting(BaseModel):
    category: str
    threshold: str


class FunctionCall(BaseModel):
    name: str
    args: dict[str, Any]
    
    model_config = ConfigDict(extra="allow")

class FunctionResponse(BaseModel):
    name: str
    response: dict[str, Any]
    
    model_config = ConfigDict(extra="allow")

class InlineData(BaseModel):
    mimeType: str = Field(alias="mimeType")
    data: str
    
    model_config = ConfigDict(populate_by_name=True)

class FileData(BaseModel):
    mimeType: str = Field(alias="mimeType")
    fileUri: str = Field(alias="fileUri")
    
    model_config = ConfigDict(populate_by_name=True)

class ContentPart(BaseModel):
    text: str | None = None
    functionCall: FunctionCall | None = Field(None, alias="functionCall")
    functionResponse: FunctionResponse | None = Field(None, alias="functionResponse")
    inlineData: InlineData | None = Field(None, alias="inlineData")
    fileData: FileData | None = Field(None, alias="fileData")
    thought: str | bool | None = None
    thoughtSignature: str | None = Field(None, alias="thoughtSignature")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class Content(BaseModel):
    parts: list[ContentPart]
    role: str | None = None
    
    model_config = ConfigDict(extra="allow")

class FunctionDeclaration(BaseModel):
    name: str
    description: str | None = None
    parameters: JSONSchema | None = None
    
    model_config = ConfigDict(extra="allow")

class Tool(BaseModel):
    functionDeclarations: list[FunctionDeclaration] = Field(alias="functionDeclarations")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class FunctionCallingConfig(BaseModel):
    mode: str | None = None
    allowedFunctionNames: list[str] | None = Field(None, alias="allowedFunctionNames")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class ToolConfig(BaseModel):
    functionCallingConfig: FunctionCallingConfig | None = Field(None, alias="functionCallingConfig")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class SystemInstruction(BaseModel):
    parts: list[ContentPart]
    
    model_config = ConfigDict(extra="allow")

class SafetyRating(BaseModel):
    category: str
    probability: str
    blocked: bool | None = None
    
    model_config = ConfigDict(extra="allow")

class CitationSource(BaseModel):
    startIndex: int | None = Field(None, alias="startIndex")
    endIndex: int | None = Field(None, alias="endIndex")
    uri: str | None = None
    license: str | None = None
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class CitationMetadata(BaseModel):
    citationSources: list[CitationSource] | None = Field(None, alias="citationSources")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class GroundingChunk(BaseModel):
    web: dict[str, str] | None = None
    retrievedContext: dict[str, str] | None = Field(None, alias="retrievedContext")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class GroundingSupport(BaseModel):
    segment: dict[str, int | str] | None = None
    groundingChunkIndices: list[int] | None = Field(None, alias="groundingChunkIndices")
    confidenceScores: list[float] | None = Field(None, alias="confidenceScores")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class SearchEntryPoint(BaseModel):
    renderedContent: str | None = Field(None, alias="renderedContent")
    sdkBlob: str | None = Field(None, alias="sdkBlob")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class RetrievalMetadata(BaseModel):
    googleSearchDynamicRetrievalScore: float | None = Field(None, alias="googleSearchDynamicRetrievalScore")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class GroundingMetadata(BaseModel):
    groundingChunks: list[GroundingChunk] | None = Field(None, alias="groundingChunks")
    groundingSupports: list[GroundingSupport] | None = Field(None, alias="groundingSupports")
    webSearchQueries: list[str] | None = Field(None, alias="webSearchQueries")
    searchEntryPoint: SearchEntryPoint | None = Field(None, alias="searchEntryPoint")
    retrievalMetadata: RetrievalMetadata | None = Field(None, alias="retrievalMetadata")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class UsageMetadata(BaseModel):
    promptTokenCount: int | None = Field(None, alias="promptTokenCount")
    candidatesTokenCount: int | None = Field(None, alias="candidatesTokenCount")
    totalTokenCount: int | None = Field(None, alias="totalTokenCount")
    cachedContentTokenCount: int | None = Field(None, alias="cachedContentTokenCount")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class PromptFeedback(BaseModel):
    blockReason: str | None = Field(None, alias="blockReason")
    safetyRatings: list[SafetyRating] | None = Field(None, alias="safetyRatings")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class Candidate(BaseModel):
    content: Content | None = None
    finishReason: str | None = Field(None, alias="finishReason")
    index: int | None = None
    safetyRatings: list[SafetyRating] | None = Field(None, alias="safetyRatings")
    citationMetadata: CitationMetadata | None = Field(None, alias="citationMetadata")
    groundingMetadata: GroundingMetadata | None = Field(None, alias="groundingMetadata")
    tokenCount: int | None = Field(None, alias="tokenCount")
    avgLogprobs: float | None = Field(None, alias="avgLogprobs")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class GeminiResponse(BaseModel):
    candidates: list[Candidate] | None = None
    promptFeedback: PromptFeedback | None = Field(None, alias="promptFeedback")
    usageMetadata: UsageMetadata | None = Field(None, alias="usageMetadata")
    createTime: str | None = Field(None, alias="createTime")
    modelVersion: str | None = Field(None, alias="modelVersion")
    responseId: str | None = Field(None, alias="responseId")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class GeminiPayload(BaseModel):
    """
    客户端发来的原始 Gemini API 请求载荷。
    支持 contents, tools, safetySettings, generationConfig 等标准字段。
    """
    contents: list[Content]
    tools: list[Tool] | None = None
    toolConfig: ToolConfig | None = Field(None, alias="toolConfig")
    systemInstruction: SystemInstruction | None = Field(None, alias="systemInstruction")
    safetySettings: list[SafetySetting] | None = Field(None, alias="safetySettings")
    generationConfig: GenerationConfig | None = Field(None, alias="generationConfig")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class StreamState(BaseModel):
    finish_reason: str | None = None
    safety_ratings: list[dict[str, Any]] = Field(default_factory=list)
    citation_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    grounding_metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None
    avg_logprobs: float | None = None
    candidate_index: int = 0
    create_time: str | None = None
    model_version: str | None = None
    prompt_feedback: dict[str, Any] = Field(default_factory=dict)
    response_id: str | None = None
    has_error: bool = False
    error_message: str = ""
    parts_by_path: dict[str, ContentPart] = Field(default_factory=dict)
    unindexed_parts: list[dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="allow")


class VertexVariables(BaseModel):
    model: str
    contents: list[Content]
    tools: list[Tool] | None = None
    toolConfig: ToolConfig | None = Field(None, alias="toolConfig")
    systemInstruction: SystemInstruction | None = Field(None, alias="systemInstruction")
    safetySettings: list[SafetySetting] | None = Field(None, alias="safetySettings")
    generationConfig: GenerationConfig | None = Field(None, alias="generationConfig")
    region: str | None = None
    recaptchaToken: str | None = Field(None, alias="recaptchaToken")
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class VertexRequest(BaseModel):
    querySignature: str | None = Field(None, alias="querySignature")
    operationName: str | None = Field(None, alias="operationName")
    variables: VertexVariables
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class RequestContext(BaseModel):
    downstream_payload: GeminiPayload
    upstream_payload: VertexRequest
    
    model_config = ConfigDict(extra="allow")