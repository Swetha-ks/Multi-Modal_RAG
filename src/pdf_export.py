"""
pdf_export.py — Export a chat conversation (+ recommended products)
to a downloadable PDF using fpdf2 (pure-Python, no system deps).
"""

from datetime import datetime
from fpdf import FPDF


def _clean(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("₹", "Rs.")
    return text.encode("latin-1", "replace").decode("latin-1")


class ChatPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(2, 128, 144)
        self.cell(0, 10, "ShopAI - Conversation Export", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, datetime.now().strftime("Exported on %d %b %Y, %I:%M %p"),
                  ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_chat_to_pdf(username: str, messages: list, output_path: str) -> str:
    pdf = ChatPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, _clean(f"User: {username}"), ln=True)
    pdf.ln(4)

    for msg in messages:
        role = msg.get("role", "assistant")
        content = _clean(msg.get("content", ""))
        products = msg.get("products") or []

        label = "You:" if role == "user" else "ShopAI:"
        color = (2, 128, 144) if role == "user" else (0, 150, 120)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, 7, label)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 7, content)

        if products:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(232, 249, 244)
            pdf.cell(60, 6, "Product", border=1, fill=True)
            pdf.cell(30, 6, "Price", border=1, fill=True)
            pdf.cell(40, 6, "Color", border=1, fill=True)
            pdf.cell(40, 6, "Brand", border=1, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 9)
            for p in products[:5]:
                pdf.cell(60, 6, _clean(str(p.get("name", ""))[:32]), border=1)
                pdf.cell(30, 6, _clean(f"Rs.{p.get('price', '')}"), border=1)
                pdf.cell(40, 6, _clean(str(p.get("color", ""))), border=1)
                pdf.cell(40, 6, _clean(str(p.get("brand", ""))), border=1)
                pdf.ln()

        pdf.ln(3)

    pdf.output(output_path)
    return output_path
