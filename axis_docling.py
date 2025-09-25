from docling.document_converter import DocumentConverter
import json

def clean_docling_doc(result):
    # Export as dict
    doc_dict = result.document.export_to_dict()

    # Step 1: Remove footer/header
    filtered_elements = [
        block for block in doc_dict.get("elements", [])
        if block.get("category") not in ("footer", "header")
    ]

    # Step 2: Sort blocks by page + vertical position (bbox[1])
    sorted_elements = sorted(
        filtered_elements,
        key=lambda b: (b.get("page", 0), b.get("bbox", [0, 0, 0, 0])[1])
    )

    doc_dict["elements"] = sorted_elements

    # Step 3: Overwrite internal dict and re-export
    result.document._doc_dict = doc_dict  # ⚠ private attr, but works
    return result.document.export_to_markdown()


if __name__ == "__main__":
    converter = DocumentConverter()
    source = "https://www.axisbank.com/docs/default-source/default-document-library/cashback-tncs---final.pdf"

    result = converter.convert(source)

    markdown_text = clean_docling_doc(result)

    with open("ktyskcleaned.md", "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print("✅ Cleaned Markdown saved as ktyskcleaned.md")
