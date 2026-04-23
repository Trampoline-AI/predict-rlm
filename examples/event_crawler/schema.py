from pydantic import BaseModel, Field


class EventOntology(BaseModel):
    """Defines what categories of information to extract from the event."""

    categories: list[str] = Field(
        default_factory=lambda: [
            "speakers",
            "attendees",
            "schedule",
            "sponsors",
            "organizers",
        ],
        description=(
            "Entity categories to extract from the event pages. "
            "Each category becomes a section in the executive summary."
        ),
    )
