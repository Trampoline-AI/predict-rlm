from pydantic import BaseModel, Field


class KeyDate(BaseModel):
    """A key date extracted from a document."""

    name: str = Field(description="e.g. 'Submission Deadline', 'Effective Date'")
    date: str = Field(description="ISO format date (YYYY-MM-DD)")
    time: str | None = Field(
        None, description="24-hour format (HH:MM), e.g. '14:00', '09:30'"
    )
    timezone: str | None = Field(
        None, description="Timezone code, e.g. 'EST', 'EDT', 'PST', 'UTC'"
    )


class KeyEntity(BaseModel):
    """A key entity (person, organization, or role) extracted from a document."""

    name: str = Field(description="Name of the person, organization, or role")
    role: str | None = Field(None, description="Role or relationship to the document")
    contact: str | None = Field(None, description="Contact info if available")


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document set."""

    report: str = Field(
        description="Full analysis as a well-formatted markdown report"
    )
    key_dates: list[KeyDate] = Field(
        default_factory=list, description="Important dates found in the documents"
    )
    key_entities: list[KeyEntity] = Field(
        default_factory=list,
        description="Key people, organizations, or roles mentioned in the documents",
    )
