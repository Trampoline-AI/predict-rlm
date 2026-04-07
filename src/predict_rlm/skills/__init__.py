"""Built-in skills library for predict-rlm.

Usage::

    from predict_rlm.skills import pdf, spreadsheet, docx

    rlm = PredictRLM(sig, skills=[pdf, spreadsheet, docx])
"""

from .docx import docx_skill as docx
from .pdf import pdf_skill as pdf
from .spreadsheet import spreadsheet_skill as spreadsheet

__all__ = ["docx", "pdf", "spreadsheet"]
