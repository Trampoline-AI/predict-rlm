"""Capabilities and the built-in capability library for predict-rlm.

Usage::

    from predict_rlm.capabilities import pdf, spreadsheet, docx

    rlm = PredictRLM(sig, capabilities=[pdf, spreadsheet, docx])
"""

from .core import Capability, merge_capabilities
from .docx import docx_capability as docx
from .pdf import pdf_capability as pdf
from .spreadsheet import spreadsheet_capability as spreadsheet

__all__ = ["Capability", "docx", "merge_capabilities", "pdf", "spreadsheet"]
