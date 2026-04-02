import dspy

from predict_rlm import File

from .schema import InvoiceExtractionResult


class ProcessInvoices(dspy.Signature):
    """Extract structured data from PDF invoices into an Excel spreadsheet.

    1. **Survey the invoices** — file names, page counts, vendor names visible
       on each document. Print a summary.

    2. **Render each page** as an image and use predict() to extract vendor
       info, line items, totals, and dates. Process multiple pages in parallel
       with asyncio.gather().

    3. **Build the Excel workbook** using openpyxl:
       - Create one sheet per invoice with line items (columns: Description,
         Quantity, Unit Price, Amount)
       - Add a "Summary" sheet with one row per invoice (columns: Vendor,
         Invoice #, Date, Due Date, Subtotal, Tax, Total)
       - Auto-size all columns for readability

    4. **Save the workbook** to the output file path.

    5. **Produce the result** with extracted invoice data, combined total,
       and a summary of what was processed.
    """

    invoices: list[File] = dspy.InputField(
        desc="PDF invoice files to process"
    )
    workbook: File = dspy.OutputField(
        desc="Excel workbook file (.xlsx) with extracted invoice data"
    )
    result: InvoiceExtractionResult = dspy.OutputField(
        desc="Structured extraction result with invoice details and totals"
    )
