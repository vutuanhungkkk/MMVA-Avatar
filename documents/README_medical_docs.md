# Medical Documents for RAG Knowledge Base

Upload PDF/TXT/DOCX files into this directory. The Health AI Assistant will
automatically index them and use them to answer health-related questions.

## Recommended Document Types

| Category | Examples |
|---|---|
| 🩹 First Aid Manual | WHO First Aid Guide, Red Cross Handbook |
| 💊 Drug Reference | BNF (British National Formulary), MIMS drug database exports |
| 🥗 Nutrition Guidelines | WHO dietary guidelines, clinical nutrition handbooks |
| 🏥 Clinical Guidelines | WHO/MOH treatment protocols, disease management guides |
| 📊 Lab Reference Ranges | Normal ranges for blood tests, urinalysis, etc. |
| 📝 Patient Education | Diabetes management, hypertension care sheets |

## How It Works

1. Place your `.pdf`, `.txt`, `.md`, `.csv`, or `.docx` files in this directory
2. Start the Health AI Assistant (click "Apply & Start" in the UI)
3. The system will automatically chunk and index all documents
4. Ask questions in the chat — the AI will search your documents first

## Configuration

Default chunking settings (adjustable in `.env`):
- `RAG_CHUNK_SIZE=1500` — Characters per chunk (larger for medical tables)
- `RAG_CHUNK_OVERLAP=300` — Overlap between chunks for context continuity
- `RAG_NUM_RETRIEVED=5` — Number of chunks retrieved per query

## Notes

- Larger documents (>100 pages) may take a few seconds to index
- For best results, use well-formatted PDFs with selectable text (not scanned images)
- You can upload additional documents at any time via the "📋 Medical Record" button in the UI
