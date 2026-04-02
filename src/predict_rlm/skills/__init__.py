"""Built-in skills library for predict-rlm.

Usage::

    from predict_rlm.skills import pdf, spreadsheet

    rlm = PredictRLM(sig, skills=[pdf, spreadsheet])
"""

from .pdf import pdf_skill as pdf
from .spreadsheet import spreadsheet_skill as spreadsheet

__all__ = ["pdf", "spreadsheet"]
