from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single line item from an invoice."""

    description: str = Field(description="Description of the item or service")
    quantity: float = Field(description="Quantity of items")
    unit_price: float = Field(description="Price per unit in dollars")
    amount: float = Field(description="Total amount for this line item in dollars")


class Invoice(BaseModel):
    """Structured data extracted from a single invoice."""

    vendor_name: str = Field(description="Name of the vendor/supplier")
    invoice_number: str = Field(description="Invoice number or reference")
    date: str = Field(description="Invoice date in ISO format (YYYY-MM-DD)")
    due_date: str = Field(description="Payment due date in ISO format (YYYY-MM-DD)")
    subtotal: float = Field(description="Subtotal before tax and discounts")
    tax: float = Field(description="Tax amount in dollars")
    total: float = Field(description="Total amount due in dollars")
    line_items: list[LineItem] = Field(description="Individual line items from the invoice")


class InvoiceExtractionResult(BaseModel):
    """Result of processing multiple invoices."""

    invoices: list[Invoice] = Field(description="Extracted data from each invoice")
    total_amount: float = Field(
        description="Combined total across all invoices in dollars"
    )
    summary: str = Field(
        description="Brief summary of the invoices processed and workbook organization"
    )
