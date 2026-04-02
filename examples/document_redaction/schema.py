from pydantic import BaseModel, Field


class RedactionTarget(BaseModel):
    """A specific piece of text identified for redaction."""

    page: int = Field(description="0-indexed page number where the text appears")
    text: str = Field(
        description="Exact text string to redact as it appears in the document"
    )
    category: str = Field(
        description="Category of sensitive information, e.g. 'person_name', "
        "'phone_number', 'address', 'email', 'ssn', 'account_number', 'custom'"
    )
    reason: str = Field(description="Why this text should be redacted")


class PageRedactionSummary(BaseModel):
    """Summary of redactions applied to a single page."""

    page: int = Field(description="0-indexed page number")
    redaction_count: int = Field(description="Number of redactions applied on this page")
    categories: list[str] = Field(
        default_factory=list,
        description="Distinct categories of redacted content on this page",
    )


class RedactionResult(BaseModel):
    """Result of the document redaction process."""

    total_redactions: int = Field(description="Total number of redactions applied")
    page_summaries: list[PageRedactionSummary] = Field(
        default_factory=list,
        description="Per-page summary of redactions applied",
    )
    targets: list[RedactionTarget] = Field(
        default_factory=list,
        description="All redaction targets identified across the document",
    )
