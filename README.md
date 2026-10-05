# Anarkali

**A small, fast decision model for web data work: scraping, crawling, and price, SEO and competitor monitoring.**

Anarkali sits in front of your scraper or your LLM pipeline and answers small typed questions about a web page:
*what kind of page is this? which text is the price? is it in stock?* It runs on a CPU, returns a probability
for every answer, and abstains when it is unsure, so only the hard pages go to an LLM or a person.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)

## Status

| Workflow | Decisions | Status |
|---|---|---|
| Site scrape | `page_type`, `price_field`, `in_stock` | training (v0) |
| Price monitoring | `price_field`, `in_stock` | training (v0) |
| Competitor monitoring | `change_type`, `alert_worthy` | planned |
| SEO monitoring | `keyword_intent`, `seo_impact` | planned |
| Site crawl | `follow_link`, `recrawl_priority` | planned |

No web-decision results are published yet. Numbers will appear here only once the evaluation in this repo produces them.
`release-150m/` is the earlier general-purpose model. It was not trained for web pages.

## Train

Open [`Anarkali_Web_Train.ipynb`](Anarkali_Web_Train.ipynb) on Colab or Kaggle with a GPU and run all cells. It will:

1. stream real pages from Common Crawl;
2. label them from each page's own schema.org markup (the model never sees that markup);
3. split by website, so the test split only has sites the model never saw;
4. train the packed Ettin-150M encoder;
5. score it against baselines, including accuracy when the model is at least 0.9 confident;
6. export ONNX and save `best.pt`.

Or run the steps yourself:

```bash
pip install -e ".[train,onnx]" warcio
python scripts/build_web_decisions.py --warcs 4 --output artifacts/web-decisions-v0
python scripts/train_anarkali.py --data artifacts/web-decisions-v0 --output artifacts/web-run \
  --architecture packed --model jhu-clsp/ettin-encoder-150m --selection-metric accuracy_then_ce
python scripts/evaluate_checkpoint.py --checkpoint artifacts/web-run/best.pt --data artifacts/web-decisions-v0 --device cuda
python scripts/export_onnx.py --checkpoint artifacts/web-run/best.pt --output artifacts/release-web --data artifacts/web-decisions-v0
```

## Use

```python
from anarkali import Engine
from anarkali.web import page_state, PAGE_TYPE_QUESTION, IN_STOCK_QUESTION

engine = Engine.load("artifacts/release-web")      # an exported model directory (ONNX, CPU only)
state = page_state(html, url)                       # always build the input with page_state
result = engine.predict(state, {"page_type": PAGE_TYPE_QUESTION, "in_stock": IN_STOCK_QUESTION})
print(result["answers"]["page_type"])               # choice, a probability per option, and confidence
```

From the command line, or as an HTTP API:

```bash
anarkali decide --model artifacts/release-web --request examples/requests/web_product_page.json
anarkali serve --model artifacts/release-web --port 8000      # POST /v1/systemone; set ANARKALI_API_KEY
docker compose up --build -d                                  # same API in a container
```

Question types:

| Type | Use it for | Answer |
|---|---|---|
| `choice` | pick one of named options | `choice` + a probability per option |
| `noul` | true or false | `noul` = probability of true |
| `score` | pick a level on an ordered scale | `score` = expected level + a probability per level |

## Repo layout

| Path | What it is |
|---|---|
| `Anarkali_Web_Train.ipynb` | the training notebook |
| `src/anarkali/web.py` | HTML to model input, and schema.org labels |
| `src/anarkali/engine.py`, `server.py`, `cli.py` | inference: Python, HTTP API, command line |
| `src/anarkali/packed.py`, `objectives.py` | model and training losses |
| `scripts/` | dataset builder, training, evaluation, ONNX export |
| `tests/` | `python -m unittest discover -s tests` |
| `CLAUDE.md` | product goal, quality bar and working rules |

## Credits

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Built on the [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-150m) encoder (JHU CLSP, MIT). Web pages come from [Common Crawl](https://commoncrawl.org). The optional mix-in data is [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0). The `/v1/systemone` format follows Jev's public API.
