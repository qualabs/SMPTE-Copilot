

## 1. Parsing Technologies

Parsing technologies are essential for transforming raw, unstructured files into clean, structured data (often Markdown or JSON) that is ready for chunking and vector indexing.

---

### 1.1. PyMuPDF4LLM

PyMuPDF4LLM is a high-level Python tool built on top of PyMuPDF, specifically designed to prepare documents for LLM and RAG workflows.

* **Core Functionality:** Converts various file types, especially **PDFs**, directly into Markdown (MD) format, which is highly compatible and easily understood by LLMs.
* **RAG Features:** It detects and formats headers, lists, bold/italic text, and code blocks. It supports extracting content from multi-column pages and integrating image references within the MD text. 
* **Reference Link:** [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)

### 1.2. Docling

Docling is an open-source document processing toolkit aimed at providing precise document parsing and structuring for the GenAI ecosystem.

* **Core Functionality:** Parses a wide range of formats (PDF, DOCX, PPTX, HTML, audio, images). It preserves structural information like page layout, reading order, and table structures.
* **RAG Features:** It exports structured data as Markdown, HTML, or JSON. It supports multimodal inputs (transcribing audio and generating image captions using vision models) and offers hierarchical chunking mechanisms. It can be run locally for sensitive data environments.
* **Reference Link:** [Docling Project GitHub](https://github.com/docling-project/docling)

### 1.3. Unstructured.io

Unstructured.io is a robust solution designed to preprocess and standardize data from over 25 different unstructured file types into a clean JSON structure. 
- **Core Functionality:** Preprocesses and standardizes data from over 25 different unstructured file types into a clean structure of 'Elements' (Title, NarrativeText, ListItem, etc.), which is typically outputted in **JSON** format. 
- **RAG Features:** It offers dedicated chunking strategies like `basic` and `by_title`, which aim to split text at section or page breaks to create more structured chunks. It is highly useful for managing and cleaning diverse data sources before vectorization. 
- **Reference Link:**[Unstructured.io Official Website](https://unstructured.io/)

---

## 2. Chunking Strategies: The 5 Levels of Text Splitting

Text splitting, or **chunking**, is a  step in a Retrieval-Augmented Generation (RAG) pipeline to break down large documents into smaller, semantically meaningful units.

### Levels of Chunking 

| Level       | Strategy                        | Description                                                                                                                               | Key Focus                                 |
| :---------- | :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- |
| **Level 1** | **Character Splitting**         | Simple static split based on a fixed number of characters, regardless of content or structure.                                            | Predictable size, minimal overhead.       |
| **Level 2** | **Recursive Splitting**         | Splits text recursively using a list of separators (e.g., `\n\n`, `\n`, `.`, ` `) to maintain paragraph integrity and structure.          | Context preservation based on structure.  |
| **Level 3** | **Document-Specific Splitting** | Tailored chunking methods optimized for specific document types like PDF, Markdown, or code files (e.g., splitting Markdown by `#` tags). | Optimizes for file format structure.      |
| **Level 4** | **Semantic Splitting**          | Goes beyond physical structure, relying on sentence embeddings to cluster text chunks based on **meaning** or semantic similarity.        | Coherence and semantic integrity.         |
| **Level 5** | **Agentic Splitting**           | An experimental, cutting-edge method that uses an agent-like system to intelligently organize and split content.                          | Intelligence and human-like organization. |
