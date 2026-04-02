from typing import Literal

from pydantic import BaseModel, Field


class SectionDiff(BaseModel):
    """Comparison of a specific section between two documents."""

    section_name: str = Field(description="Name or title of the section being compared")
    document_a_text: str = Field(description="Key text or summary from Document A")
    document_b_text: str = Field(description="Key text or summary from Document B")
    difference_summary: str = Field(description="Summary of what changed between the versions")
    significance: Literal["major", "minor", "identical"] = Field(
        description="How significant the difference is"
    )


class KeyDifference(BaseModel):
    """A high-level difference between the contracts."""

    area: str = Field(description="Area of the contract affected (e.g. pricing, liability)")
    description: str = Field(description="Description of the difference")
    impact: str = Field(description="Potential impact or implication of this difference")


class ComparisonResult(BaseModel):
    """Result of comparing two or more contracts."""

    report: str = Field(description="Full comparison report in markdown format")
    section_diffs: list[SectionDiff] = Field(
        description="Per-section comparison between the documents"
    )
    key_differences: list[KeyDifference] = Field(
        description="High-level key differences with impact analysis"
    )
    summary: str = Field(
        description="Executive summary of the most important differences"
    )
