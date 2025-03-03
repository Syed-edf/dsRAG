# dsParse
dsParse is a sub-module of dsRAG that does multimodal file parsing, semantic sectioning, and chunking. You provide a file path (and some config params) and receive nice clean chunks.

# dsRAG
[![Discord](https://img.shields.io/discord/1234629280755875881.svg?label=Discord&logo=discord&color=7289DA)](https://discord.gg/NTUVX9DmQ3)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://d-star-ai.github.io/dsRAG/)

The two creators of dsRAG, Zach and Nick McCormick, run a small applied AI consulting firm. We specialize in building high-performance RAG-based applications (naturally). As former startup founders and YC alums, we bring a business and product-centric perspective to the projects we work on. We do a mix of advisory and implementation work. If you'd like to hire us, fill out this [form](https://forms.gle/zbQwDJp7pBQKtqVT8) and we'll be in touch.

## What is dsRAG?
dsRAG is a retrieval engine for unstructured data. It is especially good at handling challenging queries over dense text, like financial reports, legal documents, and academic papers. dsRAG achieves substantially higher accuracy than vanilla RAG baselines on complex open-book question answering tasks. On one especially challenging benchmark, [FinanceBench](https://arxiv.org/abs/2311.11944), dsRAG gets accurate answers 96.6% of the time, compared to the vanilla RAG baseline which only gets 32% of questions correct.

There are three key methods used to improve performance over vanilla RAG systems:
1. Semantic sectioning
2. AutoContext
3. Relevant Segment Extraction (RSE)

#### Semantic sectioning
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting and ending lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM is also prompted to generate descriptive titles for each section. These section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

#### AutoContext (contextual chunk headers)
AutoContext creates contextual chunk headers that contain document-level and section-level context, and prepends those chunk headers to the chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

#### Relevant Segment Extraction
Relevant Segment Extraction (RSE) is a query-time post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

For example, suppose you have a bunch of SEC filings in a knowledge base and you ask “What were Apple’s key financial results in the most recent fiscal year?” RSE will identify the most relevant segment as the entire “Consolidated Statement of Operations” section, which will be 5-10 chunks long. Whereas if you ask “Who is Apple’s CEO?” the most relevant segment will be identified as a single chunk that mentions “Tim Cook, CEO.”

# Eval results
We've evaluated dsRAG on a couple of end-to-end RAG benchmarks.

#### FinanceBench
First, we have [FinanceBench](https://arxiv.org/abs/2311.11944). This benchmark uses a corpus of a few hundred 10-Ks and 10-Qs. The queries are challenging, and often require combining multiple pieces of information. Ground truth answers are provided. Answers are graded manually on a pass/fail basis. Minor allowances for rounding errors are allowed, but other than that the answer must exactly match the ground truth answer to be considered correct.

The baseline retrieval pipeline, which uses standard chunking and top-k retrieval, achieves a score of **19%** according to the paper, and **32%** according to our own experiment, which uses updated embedding and response models. dsRAG, using mostly default parameters and Claude 3.5 Sonnet (10-22-2024 version) for response generation, achieves a score of **96.6%**.

#### KITE
We couldn't find any other suitable end-to-end RAG benchmarks, so we decided to create our own, called [KITE](https://github.com/D-Star-AI/KITE) (Knowledge-Intensive Task Evaluation).

KITE currently consists of 4 datasets and a total of 50 questions.
- **AI Papers** - ~100 academic papers about AI and RAG, downloaded from arXiv in PDF form.
- **BVP Cloud 10-Ks** - 10-Ks for all companies in the Bessemer Cloud Index (~70 of them), in PDF form.
- **Sourcegraph Company Handbook** - ~800 markdown files, with their original directory structure, downloaded from Sourcegraph's publicly accessible company handbook GitHub [page](https://github.com/sourcegraph/handbook/tree/main/content).
- **Supreme Court Opinions** - All Supreme Court opinions from Term Year 2022 (delivered from January '23 to June '23), downloaded from the official Supreme Court [website](https://www.supremecourt.gov/opinions/slipopinion/22) in PDF form.

Ground truth answers are included with each sample. Most samples also include grading rubrics. Grading is done on a scale of 0-10 for each question, with a strong LLM doing the grading.

We tested four configurations:
- Top-k retrieval (baseline)
- Relevant segment extraction (RSE)
- Top-k retrieval with contextual chunk headers (CCH)
- CCH+RSE (dsRAG default config, minus semantic sectioning)

Testing RSE and CCH on their own, in addition to testing them together, lets us see the individual contributions of those two features.

Cohere English embeddings and the Cohere 3 English reranker were used for all configurations. LLM responses were generated with GPT-4o, and grading was also done with GPT-4o.

|                         | Top-k    | RSE    | CCH+Top-k    | CCH+RSE    |
|-------------------------|----------|--------|--------------|------------|
| AI Papers               | 4.5      | 7.9    | 4.7          | 7.9        |
| BVP Cloud               | 2.6      | 4.4    | 6.3          | 7.8        |
| Sourcegraph             | 5.7      | 6.6    | 5.8          | 9.4        |
| Supreme Court Opinions  | 6.1      | 8.0    | 7.4          | 8.5        |
| **Average**             | 4.72     | 6.73   | 6.04         | 8.42       |

Using CCH and RSE together leads to a dramatic improvement in performance, from 4.72 -> 8.42. Looking at the RSE and CCH+Top-k results, we can see that using each of those features individually leads to a large improvement over the baseline, with RSE appearing to be slightly more important than CCH.

Note: we did not use semantic sectioning for any of the configurations tested here. We'll evaluate that one separately once we finish some of the improvements we're working on for it. We also did not use AutoQuery, as the KITE questions are all suitable for direct use as search queries.

# Tutorial

#### Installation
To install the python package, run
```console
pip install dsrag
```

#### Quickstart
By default, dsRAG uses OpenAI for embeddings and AutoContext, and Cohere for reranking, so to run the code below you'll need to make sure you have API keys for those providers set as environmental variables with the following names: `OPENAI_API_KEY` and `CO_API_KEY`. **If you want to run dsRAG with different models, take a look at the "Basic customization" section below.**

You can create a new KnowledgeBase directly from a file using the `create_kb_from_file` helper function:
```python
sections, chunks = parse_and_chunk(
    kb_id = "sample_kb",
    doc_id = "sample_doc",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "gemini",
            "model": "gemini-1.5-pro-002",
        }
    }
    file_path="path/to/file.pdf",
)
```

dsParse can be used on its own, as shown above, or in conjunction with a dsRAG knowledge base. To use it with dsRAG, you just use the `add_document` function like normal, but set `use_vlm` to True in the `file_parsing_config` dictionary, and include a `vlm_config`.

```python
kb = KnowledgeBase(kb_id="mck_energy_test")
kb.add_document(
    doc_id="mck_energy_report",
    file_path=file_path,
    document_title="McKinsey Energy Report",
    file_parsing_config={
        "use_vlm": True,
        "vlm_config": {
            "provider": "vertex_ai",
            "model": "gemini-1.5-pro-002",
            "project_id": os.environ["VERTEX_PROJECT_ID"],
            "location": "us-central1",
        }
    }
)
```

## Installation
If you want to use dsParse on its own, without installing the full `dsrag` package, there is a standalone Python package available for dsParse, which can be installed with `pip install dsparse`. If you already have `dsrag` installed, you DO NOT need to separately install `dsparse`.

To use the VLM file parsing functionality, you'll need to install one external dependency: poppler. This is used to convert PDFs to page images. On a Mac you can install it with `brew install poppler`.

## Multimodal file parsing
dsParse uses a vision language model (VLM) to parse documents. This has a few advantages:
- It can provide descriptions for visual elements, like images and figures.
- It can parse documents that don't have extractable text (i.e. those that require OCR).
- It can accurately parse documents with complex structures.
- It can accurately categorize page content into element types.

When it comes across an element on the page that can't be accurately represented with text alone, like an image or figure (chart, graph, diagram, etc.), it provides a text description of it. This can then be used in the embedding and retrieval pipeline. 

The default model, `gemini-1.5-flash-002`, is a fast and cost-effective option. `gemini-1.5-pro-002` is also supported, and works extremely well, but at a higher cost. These models can be accessed through either the Gemini API or the Vertex API.

### Element types
Page content is categorized into the following eight categories by default:
- NarrativeText
- Figure
- Image
- Table
- Header
- Footnote
- Footer
- Equation

You can also choose to define your own categories and the VLM will be prompted accordingly.

You can choose to exclude certain element types. By default, Header and Footer elements are excluded, as they rarely contain valuable information and they break up the flow between pages. For example, if you wanted to exclude footnotes, in addition to headers and footers, you would do: `exclude_elements = ["Header", "Footer", "Footnote"]`.

## Using page images for full multimodal RAG functionality
While modern VLMs, like Gemini and Claude 3.5, are now better than traditional OCR and bounding box extraction methods at converting visual elements on a page to text or bounding boxes, they still aren’t perfect. For fully visual elements, like images or charts, getting an accurate bounding box that includes all necessary surrounding context, like legends and axis titles, is only about 90% reliable with even the best VLM models. For semi-visual content, like tables and equations, converting to plain text is also not quite perfect yet. The problem with errors at the file parsing stage is that they propagate all the way to the generation stage.

For all of these element types, it’s more reliable to just send in the original page images to the generative model as context. That ensures that no context is lost, and that OCR and other parsing errors don’t propagate to the final response generated by the model. Images are no more expensive to process than extracted text (with the exception of a few models, like GPT-4o Mini, with weird image input pricing). In fact, for pages with dense text, a full page image might actually be cheaper than using the text itself.

## Semantic sectioning and chunking
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM also generates descriptive titles for each section. When using dsParse with a dsRAG knowledge base, these section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

The default model for semantic sectioning is `gpt-4o-mini`, but similarly strong models like `gemini-1.5-flash-002` will also work well.

## Cost and latency/throughput estimation

### VLM file parsing
An obvious concern with using a large model like `gemini-1.5-pro-002` to parse documents is the cost. Let's run the numbers:

VLM file parsing cost calculation (`gemini-1.5-pro-002`)
- Image input: 1 image x $0.00032875 per image = $0.00032875
- Text input (prompt): 400 tokens x $1.25/10^6 per token = $0.000500
- Text output: 600 tokens x $5.00/10^6 per token = $0.003000
- Total: $0.00382875/page or **$3.83 per 1000 pages**

This is actually cheaper than most commercially available PDF parsing services. Unstructured and Azure Document Intelligence, for example, both cost $10 per 1000 pages. 

What about `gemini-1.5-flash-002`? Running the same calculation as above with the Gemini 1.5 Flash pricing gives a cost of **$0.23 per 1000 pages**. This is far cheaper than any commercially available OCR/PDF parsing service.

What about latency and throughput? Since each page is processed independently, this is a highly parallelizable problem. The main limiting factor then is the rate limits imposed by the VLM provider. The current rate limit for `gemini-1.5-pro-002` is 1000 requests per minute. Since dsParse uses one request per page, that means the limit is 1000 pages per minute. Processing a single page takes around 15-20 seconds, so that's the minimum latency for processing a document.

### Semantic sectioning
Semantic sectioning uses a much cheaper model, and it also uses far fewer output tokens, so it ends up being far cheaper than the file parsing step.

Semantic sectioning cost calculation (`gpt-4o-mini`)
- Input: 800 tokens x $0.15/10^6 per token = $0.00012
- Output: 50 tokens x $0.60/10^6 per token = $0.00003
- Total: $0.00015/page or **$0.15 per 1000 pages**

Document text is processed in ~5000 token mega-chunks, which is roughly ten pages on average. But these mega-chunks have to be processed sequentially for each document. Processing each mega-chunk only takes a couple seconds, though, so even a large document of a few hundred pages will only take 20-60 seconds. Rate limits for the OpenAI API are heavily dependent on the usage tier you're in.
