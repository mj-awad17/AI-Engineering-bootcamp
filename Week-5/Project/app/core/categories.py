from enum import Enum


class TicketCategory(str, Enum):
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING_PAYMENTS = "Billing & Payments"
    ACCOUNT_ACCESS = "Account & Access"
    PRODUCT_SUPPORT = "Product Support"
    SALES_PRE_SALES = "Sales & Pre-Sales"
    RETURNS_REFUNDS = "Returns & Refunds"
    CUSTOMER_SERVICE = "Customer Service"
    IT_INFRASTRUCTURE = "IT & Infrastructure"
    SERVICE_OUTAGES = "Service Outages"
    GENERAL_INQUIRIES = "General Inquiries"


CATEGORY_NAMES = [category.value for category in TicketCategory]

