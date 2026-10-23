# 🔍 Automated Plagiarism Detection and Similarity Analysis

### 🚀 AI-Powered Text Similarity Screening Using Python, NLP & Web Search

---

## 🌟 Overview

**Automated Plagiarism Detection and Similarity Analysis** is a web-based application that helps identify potentially matching text in publicly accessible web content.

Built using **Python, Flask, Natural Language Processing (NLP), and the Tavily Search API**, the application allows users to paste text into a browser-based interface and analyze it without uploading a separate reference document.

The system searches for potential source material, retrieves relevant web content, compares textual passages, and generates an estimated similarity report with source references.

🎯 **Our goal:** Make preliminary text-similarity screening simple, accessible, and informative.

## ✨ Key Features

- 📝 **Simple Text Input** — Paste text directly into the web interface.
- 🌐 **Automated Web Search** — Search publicly retrievable web content using Tavily.
- 🔎 **Exact Match Detection** — Identify consecutive matching word sequences.
- 🔗 **Partial Match Detection** — Find substantial overlapping word sequences.
- 🧠 **TF-IDF Analysis** — Rank potentially relevant source passages.
- 📊 **Similarity Percentage** — Calculate estimated matched-text coverage.
- 📚 **Source Breakdown** — Display source URLs associated with detected matches.
- 🔍 **Matching Passage Display** — Compare submitted text with retrieved source content.
- ⚡ **Interactive Interface** — Live word counter and analysis-loading indicator.
- 🔐 **Environment-Based Configuration** — Store API credentials in a local `.env` file.

## 🖥️ Application Workflow

```text
          📝 User Pastes Text
                  │
                  ▼
         🧹 Text Preprocessing
                  │
                  ▼
        ✂️ Sentence Segmentation
                  │
                  ▼
        🔎 Search Phrase Generation
                  │
                  ▼
            🌐 Tavily Search
                  │
                  ▼
          📚 Candidate Sources
                  │
                  ▼
          📥 Source Extraction
                  │
                  ▼
          🧩 Source Text Chunks
                  │
                  ▼
         ┌────────┴─────────┐
         ▼                  ▼
   🔍 Exact Matching   🔗 Partial Matching
         │                  │
         └────────┬─────────┘
                  ▼
        🧠 TF-IDF Candidate Ranking
                  │
                  ▼
       📊 Matched Word Coverage
                  │
                  ▼
          📋 Similarity Report
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Core application logic |
| 🌶️ Flask | Backend web framework |
| 🧠 NLP | Text preprocessing and comparison |
| 🌐 Tavily Search API | Discover potentially relevant web sources |
| 📥 Tavily Extract API | Retrieve readable web-page content |
| 📊 Scikit-learn | TF-IDF vectorization and cosine similarity |
| 🔗 SequenceMatcher | Find matching word sequences |
| 🎨 HTML5 & CSS3 | Web interface and styling |
| ⚡ JavaScript | Word counting and interactive UI |
| 🔌 Requests | Communicate with external APIs |
| 🔑 python-dotenv | Load local environment configuration |

## 🧠 How Does It Work?

### 1️⃣ Text Input
The user pastes the text they want to analyze into the web interface.

### 2️⃣ Text Preprocessing
The system normalizes the text and separates it into sentences and word sequences.

### 3️⃣ Search Phrase Generation
Distinctive phrases are selected to improve the likelihood of finding relevant public web sources.

### 4️⃣ Web Search
The phrases are submitted to the Tavily Search API. The application collects candidate URLs, titles, relevance information, and text snippets.

### 5️⃣ Source Extraction
Selected web pages are retrieved using the Tavily Extract API. Search snippets can serve as a fallback when full-page extraction is unavailable.

### 6️⃣ Candidate Ranking
TF-IDF and cosine similarity help identify potentially relevant source passages. These measures are used for candidate selection rather than being treated as proof of plagiarism.

### 7️⃣ Exact and Partial Matching
The system compares submitted word sequences with retrieved source text and identifies potential exact or partial matches.

### 8️⃣ Similarity Calculation
The application counts submitted-word positions covered by detected matching passages and calculates an estimated similarity percentage.

### 9️⃣ Results
The report displays similarity statistics, source references, and potentially matching passages for review.

## 📐 Similarity Calculation

The application uses the following general formula:

\[
\text{Estimated Similarity (\%)} =
\frac{\text{Matched Words}}{\text{Total Words}} \times 100
\]

**Example:**

| Metric | Example |
|---|---:|
| Total words | 1,000 |
| Matched words | 180 |
| Estimated similarity | 18% |
| Estimated unmatched content | 82% |

*The values above are illustrative, not actual measured project results.*

The application distinguishes exact and partial matches according to its implemented matching rules. Its classifications are estimates and are not equivalent to a definitive assessment of plagiarism.

## 📁 Project Structure

```text
Plagiarism-Detection/
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 .gitignore
├── 🔒 .env                 # Local only; never commit
│
├── 📁 templates/
│   └── 📄 index.html
│
└── 📁 static/
    ├── 🎨 style.css
    └── ⚡ script.js
```

### 📌 File Descriptions

| File | Description |
|---|---|
| `app.py` | Flask routes, web search, extraction, matching, and similarity calculation |
| `templates/index.html` | Text input interface and results display |
| `static/style.css` | Styles for the application |
| `static/script.js` | Live word counter and loading indicator |
| `requirements.txt` | Required Python dependencies |
| `.env` | Local API key and secret configuration |
| `.gitignore` | Prevents sensitive and temporary files from being committed |

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### ✅ Prerequisites

- Python 3.10 or a compatible Python version
- Git (optional)
- Internet connection
- A Tavily API key

### 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/IshantGarg03/Plagiarism-Detection.git
cd Plagiarism-Detection
```

Alternatively, download the repository ZIP file from GitHub and extract it.

### 🐍 Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 📦 Step 3: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 🔑 Step 4: Configure Environment Variables

Create a file named `.env` in the project root:

```env
TAVILY_API_KEY=your_tavily_api_key
FLASK_SECRET_KEY=your_private_random_secret
```

Replace the placeholders with your actual values.

⚠️ **Security reminder:** Never upload `.env` or your private API key to GitHub. Keep `.env` in `.gitignore`.

### ▶️ Step 5: Start the Application

```powershell
python app.py
```

Open the local address printed by Flask, normally:

**http://127.0.0.1:5000**

Paste your text into the text box and click **Check Plagiarism**.

## 🧪 Testing and Validation

The application should be tested with controlled examples before relying on its reported results.

| Test | Input | What to check |
|---|---|---|
| 🟢 Original text | Independently written text | Whether the system finds any unexpected matches |
| 🔴 Exact copy | Text copied from a public web page | Whether it retrieves the source and detects matching passages |
| 🟡 Mixed content | Original and copied passages | Whether it detects matching portions without counting the same submitted words twice |
| 🔁 Repeated text | A matching passage repeated in the submission | Whether duplicate submitted-word positions are counted once |
| 🔄 Paraphrased text | Rewritten source material | Whether the current lexical matching methods detect any overlap |

The actual outcome depends on search coverage, retrieved source content, and the implemented matching rules.

## ⚠️ Limitations

### 🌐 Public-Web Coverage
The application can only identify sources it successfully discovers and retrieves. It does not search every webpage on the Internet.

### 🔒 Private Academic Databases
The system does not automatically access private university repositories, subscription-only publications, or proprietary student-paper databases.

### 🔄 Paraphrasing
The current implementation primarily uses lexical and word-sequence matching. It may miss substantial paraphrasing or conceptually similar text expressed with different wording.

### 📉 Different Similarity Scores
Commercial plagiarism checkers may use different source collections, indexing infrastructure, thresholds, and algorithms. Their reported percentages may differ from this project's results.

### ⚖️ Similarity Is Not Proof of Plagiarism
A matching passage may be a quotation, a correctly cited statement, standard terminology, or independently written text. The result should be interpreted in context.

## 🚀 Future Enhancements

Potential improvements include:

- 🤖 Sentence embeddings for semantic similarity and paraphrase detection.
- 🌍 Multilingual text-similarity analysis.
- 📑 PDF and DOCX input support.
- 🖍️ Word-level highlighting of matched passages.
- 🗃️ Comparison against an authorized local document repository.
- 📚 Integration with licensed academic content sources.
- 📄 Downloadable PDF and DOCX reports.
- 🗄️ Database integration for analysis history.
- 📈 Evaluation using precision, recall, F1-score, and processing time.
- 🧠 Advanced hybrid matching using lexical and semantic methods.

## 🎓 Academic Project Information

**Project Title:** Automated Plagiarism Detection and Similarity Analysis

**Developed By:** Ishant Garg

**Program:** M.E. Artificial Intelligence

**Institution:** Chandigarh University, Punjab, India

**Purpose:** Academic project and NLP-based text-similarity screening.

## 🔐 Security and Responsible Use

- Keep API keys and secrets out of public repositories.
- Validate input and set appropriate request-size limits.
- Respect applicable API terms and website content-use policies.
- Do not treat the generated percentage as a final academic misconduct decision.
- Use a production-ready WSGI server and appropriate security controls before public deployment.

## 🤝 Contributing

Suggestions and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request with a clear explanation.

## 📜 License

No open-source license is currently specified. Add an appropriate `LICENSE` file if you intend to grant others permission to reuse, modify, or redistribute this project.

---

<p align="center">
  💡 <strong>Built to explore how NLP and public web search can support text-similarity analysis.</strong>
</p>

<p align="center">
  ⭐ If you find this project useful, consider starring the repository!
</p>
