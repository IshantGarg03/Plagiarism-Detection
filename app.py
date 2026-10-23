from collections import Counter, defaultdict
from difflib import SequenceMatcher
import math
import os
import re
from typing import Dict, List, Set, Tuple

import requests
from dotenv import load_dotenv

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for
)

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    ""
).strip()


FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY",
    "plagiarism-project-secret"
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__
)


app.secret_key = (
    FLASK_SECRET_KEY
)


# Maximum text submitted through the form.
# 2 MB is more than enough for a normal research paper.
app.config[
    "MAX_CONTENT_LENGTH"
] = 2 * 1024 * 1024


# ============================================================
# TAVILY ENDPOINTS
# ============================================================

TAVILY_SEARCH_URL = (
    "https://api.tavily.com/search"
)


TAVILY_EXTRACT_URL = (
    "https://api.tavily.com/extract"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Number of independent web-search queries.
#
# A long document might contain hundreds of sentences.
# We select the most distinctive phrases rather than
# blindly searching every sentence.
MAX_SEARCH_QUERIES = 25


# Search results per query.
RESULTS_PER_QUERY = 5


# Candidate sources to send to Tavily Extract.
#
# More sources = better recall.
MAX_SOURCES_TO_EXTRACT = 30


# Minimum submitted sentence length.
MIN_SENTENCE_WORDS = 8


# Search phrase size.
SEARCH_WINDOW_SIZE = 9


# Number of consecutive identical words considered
# strong exact-match evidence.
EXACT_NGRAM_SIZE = 6


# Minimum percentage of a submitted sentence that
# should be covered by exact matching.
MIN_EXACT_COVERAGE = 0.20


# Minimum contiguous partial matching block.
MIN_PARTIAL_BLOCK = 5


# Minimum submitted sentence coverage for partial matching.
MIN_PARTIAL_COVERAGE = 0.25


# TF-IDF candidate limit.
TOP_TFIDF_CANDIDATES = 12


# Source text chunking.
REFERENCE_CHUNK_SIZE = 120
REFERENCE_CHUNK_STEP = 60


# Never compare more than this number of source chunks.
MAX_REFERENCE_CHUNKS = 8000


# At most this many source matches for a submitted sentence.
MAX_MATCHES_PER_SENTENCE = 3


# ============================================================
# HTTP SESSION
# ============================================================

def create_http_session():
    """
    Create an HTTP session with retry support.

    Retries temporary:
        - connection errors
        - incomplete reads
        - HTTP 429
        - HTTP 5xx
    """

    session = requests.Session()


    retry_strategy = Retry(

        total=3,

        connect=3,

        read=3,

        status=3,

        backoff_factor=1,

        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],

        allowed_methods=frozenset([
            "POST"
        ]),

        raise_on_status=False

    )


    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )


    session.mount(
        "https://",
        adapter
    )


    session.mount(
        "http://",
        adapter
    )


    return session


HTTP_SESSION = (
    create_http_session()
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_spaces(
    text: str
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def normalize_text(
    text: str
) -> str:
    """
    Normalize text for comparison.

    Example:

        "Artificial Intelligence!"
    
    becomes:

        "artificial intelligence"
    """

    text = text.lower()


    # Normalize common Unicode punctuation.
    text = (
        text
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )


    # Keep only alphanumeric characters and spaces.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )


    return normalize_spaces(
        text
    )


def tokenize(
    text: str
) -> List[str]:

    return re.findall(
        r"\b[a-z0-9]+\b",
        normalize_text(text)
    )


def count_words(
    text: str
) -> int:

    return len(
        tokenize(text)
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(
    text: str
) -> List[str]:
    """
    Split the submitted text into sentences.

    Paragraphs and normal punctuation are handled.
    """

    text = text.replace(
        "\r\n",
        "\n"
    )


    text = text.replace(
        "\r",
        "\n"
    )


    blocks = re.split(
        r"\n\s*\n+",
        text
    )


    sentences = []


    for block in blocks:

        block = block.strip()


        if not block:

            continue


        parts = re.split(
            r"(?<=[.!?])\s+",
            block
        )


        for part in parts:

            part = part.strip()


            if part:

                sentences.append(
                    part
                )


    # --------------------------------------------------------
    # Handle text without punctuation.
    # --------------------------------------------------------

    if (
        len(sentences) == 1
        and count_words(
            sentences[0]
        ) > 80
    ):

        raw_words = re.findall(
            r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b",
            sentences[0]
        )


        sentences = []


        chunk_size = 30


        for start in range(
            0,
            len(raw_words),
            chunk_size
        ):

            chunk = raw_words[
                start:
                start + chunk_size
            ]


            if chunk:

                sentences.append(
                    " ".join(chunk)
                )


    return sentences


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def make_search_queries(
    sentences: List[str]
) -> List[str]:
    """
    Create distinctive 9-word phrase queries.

    We inspect several windows from every meaningful
    sentence and choose the rarest phrases.

    This is much better than searching only the first
    sentence or only the beginning of every sentence.
    """

    word_frequency = Counter()


    all_words = []


    for sentence in sentences:

        words = tokenize(
            sentence
        )


        all_words.extend(
            words
        )


        for word in set(words):

            word_frequency[
                word
            ] += 1


    document_size = max(
        len(all_words),
        1
    )


    candidates = []


    for sentence_index, sentence in enumerate(
        sentences
    ):

        raw_words = re.findall(
            r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b",
            sentence
        )


        words = [
            normalize_text(word)
            for word in raw_words
        ]


        words = [
            word
            for word in words
            if word
        ]


        if len(words) < MIN_SENTENCE_WORDS:

            continue


        window_size = min(
            SEARCH_WINDOW_SIZE,
            len(words)
        )


        for start in range(
            0,
            len(words)
            -
            window_size
            +
            1
        ):

            window = words[
                start:
                start + window_size
            ]


            rarity_score = 0.0


            for word in set(window):

                frequency = (
                    word_frequency[
                        word
                    ]
                    or
                    1
                )


                rarity_score += math.log(

                    (
                        document_size
                        +
                        1
                    )
                    /
                    (
                        frequency
                        +
                        1
                    )
                    +
                    1

                )


            # Give a small reward to complete windows.
            total_score = (
                rarity_score
                +
                0.05 * len(window)
            )


            query = (
                '"'
                +
                " ".join(window)
                +
                '"'
            )


            candidates.append(
                (

                    total_score,

                    sentence_index,

                    query

                )
            )


    # Highest rarity first.
    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True
    )


    queries = []


    seen = set()


    for (
        _score,
        _sentence_index,
        query
    ) in candidates:

        if query in seen:

            continue


        seen.add(
            query
        )


        queries.append(
            query
        )


        if (
            len(queries)
            >=
            MAX_SEARCH_QUERIES
        ):

            break


    return queries


# ============================================================
# TAVILY SEARCH
# ============================================================

def tavily_search(
    query: str
) -> List[dict]:
    """
    Search the public web.
    """

    if not TAVILY_API_KEY:

        raise RuntimeError(
            "TAVILY_API_KEY is missing in .env"
        )


    response = HTTP_SESSION.post(

        TAVILY_SEARCH_URL,

        headers={

            "Authorization":
                f"Bearer {TAVILY_API_KEY}",

            "Content-Type":
                "application/json"

        },

        json={

            "query":
                query,

            "search_depth":
                "advanced",

            "max_results":
                RESULTS_PER_QUERY,

            "include_answer":
                False,

            # We intentionally do not ask Search
            # for complete page contents.

        },

        timeout=30
    )


    if response.status_code == 401:

        raise RuntimeError(
            "Tavily API key is invalid."
        )


    if response.status_code == 429:

        raise RuntimeError(
            "Tavily API rate limit reached."
        )


    response.raise_for_status()


    data = response.json()


    return data.get(
        "results",
        []
    )


# ============================================================
# COLLECT SOURCES
# ============================================================

def collect_sources(
    queries: List[str]
) -> List[dict]:
    """
    Search all generated phrases and combine
    duplicate URLs.
    """

    sources = {}


    for number, query in enumerate(
        queries,
        start=1
    ):

        print()
        print(
            f"[SEARCH {number}/{len(queries)}]"
        )


        print(
            query
        )


        try:

            results = tavily_search(
                query
            )


        except requests.RequestException as error:

            print(
                "Network error:",
                error
            )

            continue


        print(
            "Results:",
            len(results)
        )


        for result in results:

            url = str(
                result.get(
                    "url",
                    ""
                )
            ).strip()


            if not url:

                continue


            title = str(
                result.get(
                    "title",
                    "Unknown Source"
                )
            )


            snippet = str(
                result.get(
                    "content",
                    ""
                )
            ).strip()


            score = float(
                result.get(
                    "score",
                    0
                )
                or
                0
            )


            if url not in sources:

                sources[url] = {

                    "url":
                        url,

                    "title":
                        title,

                    "score":
                        score,

                    "hits":
                        1,

                    "snippets":
                        (
                            [snippet]
                            if snippet
                            else []
                        )

                }


            else:

                sources[url][
                    "hits"
                ] += 1


                sources[url][
                    "score"
                ] = max(

                    sources[url][
                        "score"
                    ],

                    score

                )


                if snippet:

                    sources[url][
                        "snippets"
                    ].append(
                        snippet
                    )


    source_list = list(
        sources.values()
    )


    # A source returned by multiple queries is generally
    # worth inspecting first.
    source_list.sort(

        key=lambda source: (

            source["hits"],

            source["score"]

        ),

        reverse=True

    )


    return source_list


# ============================================================
# EXTRACT SOURCE CONTENT
# ============================================================

def extract_source_pages(
    sources: List[dict]
) -> List[dict]:
    """
    Extract the selected public web pages.

    If a page cannot be extracted, its search snippets
    are used as a fallback.
    """

    if not sources:

        return []


    selected = sources[
        :MAX_SOURCES_TO_EXTRACT
    ]


    urls = [
        source["url"]
        for source in selected
    ]


    title_map = {

        source["url"]:
            source["title"]

        for source
        in selected

    }


    pages = []


    extracted_urls = set()


    # --------------------------------------------------------
    # Extract in batches of <= 20.
    # --------------------------------------------------------

    for start in range(
        0,
        len(urls),
        20
    ):

        batch = urls[
            start:
            start + 20
        ]


        try:

            response = HTTP_SESSION.post(

                TAVILY_EXTRACT_URL,

                headers={

                    "Authorization":
                        f"Bearer {TAVILY_API_KEY}",

                    "Content-Type":
                        "application/json"

                },

                json={

                    "urls":
                        batch,

                    "extract_depth":
                        "advanced",

                    "include_images":
                        False

                },

                timeout=90

            )


        except requests.RequestException as error:

            print(
                "Extract network error:",
                error
            )

            continue


        if response.status_code == 401:

            raise RuntimeError(
                "Tavily API key is invalid."
            )


        if response.status_code == 429:

            raise RuntimeError(
                "Tavily Extract API rate limit reached."
            )


        response.raise_for_status()


        data = response.json()


        for result in data.get(
            "results",
            []
        ):

            url = str(
                result.get(
                    "url",
                    ""
                )
            ).strip()


            raw_content = str(
                result.get(
                    "raw_content",
                    ""
                )
            ).strip()


            if (
                not url
                or
                not raw_content
            ):

                continue


            extracted_urls.add(
                url
            )


            pages.append(
                {

                    "url":
                        url,

                    "title":
                        title_map.get(
                            url,
                            "Web Source"
                        ),

                    "text":
                        raw_content

                }
            )


    # --------------------------------------------------------
    # Fallback for failed pages.
    # --------------------------------------------------------

    for source in selected:

        url = source[
            "url"
        ]


        if url in extracted_urls:

            continue


        fallback_text = " ".join(
            source.get(
                "snippets",
                []
            )
        ).strip()


        if not fallback_text:

            continue


        pages.append(
            {

                "url":
                    url,

                "title":
                    source[
                        "title"
                    ],

                "text":
                    fallback_text

            }
        )


    return pages


# ============================================================
# SOURCE CORPUS
# ============================================================

def build_reference_chunks(
    pages: List[dict]
) -> List[dict]:
    """
    Divide source pages into overlapping chunks.

    A passage-level corpus is more appropriate for
    plagiarism detection than treating a full webpage
    as a single document.
    """

    chunks = []


    seen = set()


    for page in pages:

        words = tokenize(
            page["text"]
        )


        if len(words) < MIN_SENTENCE_WORDS:

            continue


        # ----------------------------------------------------
        # Small page.
        # ----------------------------------------------------

        if (
            len(words)
            <= REFERENCE_CHUNK_SIZE
        ):

            candidate_chunks = [
                words
            ]


        # ----------------------------------------------------
        # Large page.
        # ----------------------------------------------------

        else:

            candidate_chunks = []


            for start in range(

                0,

                len(words),

                REFERENCE_CHUNK_STEP

            ):

                candidate_chunks.append(

                    words[
                        start:
                        start
                        +
                        REFERENCE_CHUNK_SIZE
                    ]

                )


        for chunk_words in (
            candidate_chunks
        ):

            if (
                len(chunk_words)
                <
                MIN_SENTENCE_WORDS
            ):

                continue


            cleaned = " ".join(
                chunk_words
            )


            # Remove duplicate chunks.
            if cleaned in seen:

                continue


            seen.add(
                cleaned
            )


            chunks.append(
                {

                    "words":
                        chunk_words,

                    "cleaned":
                        cleaned,

                    "text":
                        cleaned,

                    "url":
                        page["url"],

                    "title":
                        page["title"]

                }
            )


            if (
                len(chunks)
                >=
                MAX_REFERENCE_CHUNKS
            ):

                return chunks


    return chunks


# ============================================================
# EXACT N-GRAM INDEX
# ============================================================

def build_ngram_index(
    chunks: List[dict]
):
    """
    Build an inverted index:

        exact 6-word phrase
             ↓
        matching source chunks
    """

    index = defaultdict(
        list
    )


    for chunk_id, chunk in enumerate(
        chunks
    ):

        words = chunk[
            "words"
        ]


        if (
            len(words)
            <
            EXACT_NGRAM_SIZE
        ):

            continue


        for start in range(

            len(words)
            -
            EXACT_NGRAM_SIZE
            +
            1

        ):

            phrase = tuple(

                words[
                    start:
                    start
                    +
                    EXACT_NGRAM_SIZE
                ]

            )


            index[
                phrase
            ].append(
                (
                    chunk_id,
                    start
                )
            )


    return index


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match_positions(
    submitted_words: List[str],
    reference_words: List[str]
) -> Tuple[Set[int], int]:
    """
    Find exact consecutive word sequences.

    Returns:

        matched submitted positions
        number of exact phrases
    """

    positions = set()


    ngram_count = 0


    if (
        len(submitted_words)
        <
        EXACT_NGRAM_SIZE

    ):

        return (
            positions,
            ngram_count
        )


    reference_ngrams = {

        tuple(
            reference_words[
                i:
                i
                +
                EXACT_NGRAM_SIZE
            ]
        )

        for i

        in range(

            len(reference_words)
            -
            EXACT_NGRAM_SIZE
            +
            1

        )

    }


    for start in range(

        len(submitted_words)
        -
        EXACT_NGRAM_SIZE
        +
        1

    ):

        phrase = tuple(

            submitted_words[
                start:
                start
                +
                EXACT_NGRAM_SIZE
            ]

        )


        if phrase not in reference_ngrams:

            continue


        ngram_count += 1


        for position in range(

            start,

            start
            +
            EXACT_NGRAM_SIZE

        ):

            positions.add(
                position
            )


    return (
        positions,
        ngram_count
    )


# ============================================================
# PARTIAL MATCH
# ============================================================

def partial_match_positions(
    submitted_words: List[str],
    reference_words: List[str]
) -> Set[int]:
    """
    Find long matching word blocks.

    Only the matched submitted-word positions
    are returned.
    """

    if (
        not submitted_words
        or
        not reference_words
    ):

        return set()


    matcher = SequenceMatcher(

        None,

        submitted_words,

        reference_words,

        autojunk=False

    )


    positions = set()


    for block in (
        matcher.get_matching_blocks()
    ):

        if (
            block.size
            >=
            MIN_PARTIAL_BLOCK
        ):

            for position in range(

                block.a,

                block.a
                +
                block.size

            ):

                positions.add(
                    position
                )


    return positions


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    submitted_words: List[str],
    reference_words: List[str]
):
    """
    Determine whether a source chunk contains
    a meaningful exact/partial match.

    IMPORTANT:

    We count matched word positions, not the
    complete sentence.
    """

    exact_positions, exact_ngrams = (
        exact_match_positions(

            submitted_words,

            reference_words

        )
    )


    partial_positions = (
        partial_match_positions(

            submitted_words,

            reference_words

        )
    )


    # Exact positions should not be counted as partial.
    partial_positions -= (
        exact_positions
    )


    matched_positions = (
        exact_positions
        |
        partial_positions
    )


    submitted_count = len(
        submitted_words
    )


    if submitted_count == 0:

        return None


    exact_count = len(
        exact_positions
    )


    partial_count = len(
        partial_positions
    )


    matched_count = len(
        matched_positions
    )


    exact_coverage = (
        exact_count
        /
        submitted_count
    )


    overall_coverage = (
        matched_count
        /
        submitted_count
    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if (
        exact_ngrams >= 1
        and
        exact_coverage
        >=
        MIN_EXACT_COVERAGE
    ):

        match_type = "exact"


    elif (
        partial_count
        >=
        MIN_PARTIAL_BLOCK
        and
        overall_coverage
        >=
        MIN_PARTIAL_COVERAGE
    ):

        match_type = "partial"


    else:

        return None


    return {

        "type":
            match_type,

        "matched_positions":
            matched_positions,

        "exact_positions":
            exact_positions,

        "partial_positions":
            partial_positions,

        "exact_words":
            exact_count,

        "partial_words":
            partial_count,

        "matched_words":
            matched_count,

        "coverage":
            round(
                overall_coverage
                *
                100,
                2
            ),

        "exact_ngrams":
            exact_ngrams

    }


# ============================================================
# TF-IDF CANDIDATE RANKING
# ============================================================

def get_tfidf_candidates(
    submitted_sentence: str,
    chunks: List[dict],
    vectorizer,
    matrix
) -> List[int]:
    """
    Use TF-IDF ONLY to find promising source chunks.

    TF-IDF itself is not treated as plagiarism.
    """

    query = normalize_text(
        submitted_sentence
    )


    if not query:

        return []


    query_vector = (
        vectorizer.transform(
            [query]
        )
    )


    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]


    indexes = (
        scores.argsort()[::-1]
    )[
        :TOP_TFIDF_CANDIDATES
    ]


    return [
        int(index)
        for index
        in indexes
    ]


# ============================================================
# ANALYZE ONE SENTENCE
# ============================================================

def analyze_sentence(
    sentence: str,
    chunks: List[dict],
    vectorizer,
    matrix,
    ngram_index
):
    """
    Find all strong matching source chunks for one
    submitted sentence.
    """

    submitted_words = tokenize(
        sentence
    )


    if (
        len(submitted_words)
        <
        MIN_SENTENCE_WORDS
    ):

        return []


    candidate_ids = set()


    # --------------------------------------------------------
    # TF-IDF candidates.
    # --------------------------------------------------------

    candidate_ids.update(

        get_tfidf_candidates(

            sentence,

            chunks,

            vectorizer,

            matrix

        )

    )


    # --------------------------------------------------------
    # Exact n-gram candidates.
    #
    # This is crucial because an exact match can be
    # recovered even when TF-IDF ranking is imperfect.
    # --------------------------------------------------------

    for start in range(

        len(submitted_words)
        -
        EXACT_NGRAM_SIZE
        +
        1

    ):

        phrase = tuple(

            submitted_words[
                start:
                start
                +
                EXACT_NGRAM_SIZE
            ]

        )


        for chunk_id, _ in (
            ngram_index.get(
                phrase,
                []
            )
        ):

            candidate_ids.add(
                chunk_id
            )


    # --------------------------------------------------------
    # Evaluate candidates.
    # --------------------------------------------------------

    results = []


    for chunk_id in candidate_ids:

        chunk = chunks[
            chunk_id
        ]


        evaluation = evaluate_candidate(

            submitted_words,

            chunk["words"]

        )


        if evaluation is None:

            continue


        sequence_score = (
            SequenceMatcher(

                None,

                submitted_words,

                chunk["words"],

                autojunk=False

            ).ratio()

        )


        evaluation[
            "chunk_id"
        ] = chunk_id


        evaluation[
            "sequence"
        ] = round(

            sequence_score
            *
            100,

            2

        )


        results.append(
            evaluation
        )


    # Highest actual matching coverage first.
    results.sort(

        key=lambda item: (

            item["matched_words"],

            item["exact_words"],

            item["coverage"],

            item["sequence"]

        ),

        reverse=True

    )


    return results[
        :10
    ]


# ============================================================
# ANALYZE COMPLETE TEXT
# ============================================================

def analyze_text(
    submitted_text: str
):
    """
    Complete plagiarism pipeline.
    """

    # --------------------------------------------------------
    # STEP 1: Sentence processing
    # --------------------------------------------------------

    sentences = split_sentences(
        submitted_text
    )


    if not sentences:

        raise ValueError(
            "No readable text was found."
        )


    print()
    print(
        "=" * 70
    )


    print(
        "AUTOMATED PLAGIARISM ANALYSIS"
    )


    print(
        "=" * 70
    )


    print(
        "Sentences:",
        len(sentences)
    )


    # --------------------------------------------------------
    # STEP 2: Search query generation
    # --------------------------------------------------------

    queries = make_search_queries(
        sentences
    )


    if not queries:

        raise ValueError(
            "Not enough meaningful text was found "
            "for web searching."
        )


    print(
        "Search queries:",
        len(queries)
    )


    # --------------------------------------------------------
    # STEP 3: Web search
    # --------------------------------------------------------

    sources = collect_sources(
        queries
    )


    if not sources:

        raise ValueError(
            "No public web sources were retrieved. "
            "The similarity percentage cannot be "
            "calculated reliably."
        )


    print(
        "Candidate sources:",
        len(sources)
    )


    # --------------------------------------------------------
    # STEP 4: Extract sources
    # --------------------------------------------------------

    pages = extract_source_pages(
        sources
    )


    if not pages:

        raise ValueError(
            "Sources were discovered, but no readable "
            "source content could be obtained."
        )


    print(
        "Pages analyzed:",
        len(pages)
    )


    # --------------------------------------------------------
    # STEP 5: Build reference chunks
    # --------------------------------------------------------

    chunks = build_reference_chunks(
        pages
    )


    if not chunks:

        raise ValueError(
            "No usable source text was obtained."
        )


    print(
        "Reference chunks:",
        len(chunks)
    )


    # --------------------------------------------------------
    # STEP 6: Build TF-IDF index
    # --------------------------------------------------------

    reference_texts = [

        chunk["cleaned"]

        for chunk
        in chunks

    ]


    vectorizer = (
        TfidfVectorizer(

            ngram_range=(1, 2),

            min_df=1,

            max_df=0.995

        )
    )


    matrix = (
        vectorizer.fit_transform(
            reference_texts
        )
    )


    # --------------------------------------------------------
    # STEP 7: Build exact-match index
    # --------------------------------------------------------

    ngram_index = build_ngram_index(
        chunks
    )


    # --------------------------------------------------------
    # STEP 8: Analyze each submitted sentence
    # --------------------------------------------------------

    raw_matches = []


    for sentence_number, sentence in enumerate(
        sentences,
        start=1
    ):

        sentence_matches = analyze_sentence(

            sentence,

            chunks,

            vectorizer,

            matrix,

            ngram_index

        )


        # Positions already covered inside this submitted
        # sentence, so the same words are not counted twice.
        used_positions = set()


        selected_matches = []


        for candidate in sentence_matches:

            new_positions = (

                candidate[
                    "matched_positions"
                ]

                -

                used_positions

            )


            if len(new_positions) < 4:

                continue


            used_positions.update(
                new_positions
            )


            chunk = chunks[
                candidate["chunk_id"]
            ]


            selected_matches.append(
                {

                    "sentence_number":
                        sentence_number,

                    "submitted_sentence":
                        sentence,

                    "reference_sentence":
                        chunk["text"],

                    "type":
                        candidate["type"],

                    "coverage":
                        candidate["coverage"],

                    "matched_words":
                        len(
                            new_positions
                        ),

                    "exact_words":
                        candidate[
                            "exact_words"
                        ],

                    "partial_words":
                        candidate[
                            "partial_words"
                        ],

                    "exact_ngrams":
                        candidate[
                            "exact_ngrams"
                        ],

                    "sequence":
                        candidate[
                            "sequence"
                        ],

                    "url":
                        chunk["url"],

                    "title":
                        chunk["title"],

                    "matched_positions":
                        new_positions

                }
            )


            if (
                len(selected_matches)
                >=
                MAX_MATCHES_PER_SENTENCE
            ):

                break


        raw_matches.extend(
            selected_matches
        )


    # --------------------------------------------------------
    # STEP 9: Global submitted-word positions
    # --------------------------------------------------------

    sentence_offsets = []


    offset = 0


    for sentence in sentences:

        sentence_offsets.append(
            offset
        )


        offset += len(
            tokenize(sentence)
        )


    exact_global = set()

    partial_global = set()


    for match in raw_matches:

        sentence_index = (
            match[
                "sentence_number"
            ]
            - 1
        )


        if (
            sentence_index
            < 0
            or
            sentence_index
            >=
            len(sentence_offsets)
        ):

            continue


        sentence_offset = (
            sentence_offsets[
                sentence_index
            ]
        )


        for local_position in (
            match[
                "matched_positions"
            ]
        ):

            global_position = (
                sentence_offset
                +
                local_position
            )


            if (
                match["type"]
                ==
                "exact"
            ):

                exact_global.add(
                    global_position
                )

            else:

                partial_global.add(
                    global_position
                )


    # Exact match takes priority.
    partial_global -= exact_global


    total_words = sum(

        len(
            tokenize(sentence)
        )

        for sentence
        in sentences

    )


    exact_words = len(
        exact_global
    )


    partial_words = len(
        partial_global
    )


    matched_words = (
        exact_words
        +
        partial_words
    )


    # --------------------------------------------------------
    # STEP 10: Percentages
    # --------------------------------------------------------

    if total_words > 0:

        exact_percentage = (

            exact_words
            /
            total_words
        ) * 100


        partial_percentage = (

            partial_words
            /
            total_words
        ) * 100


        overall_percentage = (

            matched_words
            /
            total_words
        ) * 100

    else:

        exact_percentage = 0.0

        partial_percentage = 0.0

        overall_percentage = 0.0


    unique_percentage = max(

        0.0,

        100.0
        -
        overall_percentage

    )


    # --------------------------------------------------------
    # STEP 11: Source breakdown
    # --------------------------------------------------------

    source_map = {}


    for match in raw_matches:

        url = match[
            "url"
        ]


        if url not in source_map:

            source_map[url] = {

                "title":
                    match["title"],

                "url":
                    url,

                "matched_words":
                    0,

                "matches":
                    0

            }


        source_map[url][
            "matched_words"
        ] += match[
            "matched_words"
        ]


        source_map[url][
            "matches"
        ] += 1


    source_breakdown = (
        list(
            source_map.values()
        )
    )


    for source in source_breakdown:

        if total_words:

            source[
                "percentage"
            ] = round(

                source[
                    "matched_words"
                ]
                /
                total_words
                *
                100,

                2

            )

        else:

            source[
                "percentage"
            ] = 0.0


    source_breakdown.sort(

        key=lambda source:
            source["matched_words"],

        reverse=True

    )


    # --------------------------------------------------------
    # STEP 12: Sort matches
    # --------------------------------------------------------

    raw_matches.sort(

        key=lambda match: (

            match[
                "matched_words"
            ],

            match[
                "coverage"
            ]

        ),

        reverse=True

    )


    print()
    print(
        "=" * 70
    )


    print(
        "FINAL RESULT"
    )


    print(
        "Total words:",
        total_words
    )


    print(
        "Exact matched words:",
        exact_words
    )


    print(
        "Partial matched words:",
        partial_words
    )


    print(
        "Overall similarity:",
        round(
            overall_percentage,
            2
        ),
        "%"
    )


    print(
        "Sources with matches:",
        len(source_breakdown)
    )


    print(
        "=" * 70
    )


    return {

        "percentage":
            round(
                overall_percentage,
                2
            ),

        "exact_percentage":
            round(
                exact_percentage,
                2
            ),

        "partial_percentage":
            round(
                partial_percentage,
                2
            ),

        "unique_percentage":
            round(
                unique_percentage,
                2
            ),

        "total_words":
            total_words,

        "matched_words":
            matched_words,

        "exact_words":
            exact_words,

        "partial_words":
            partial_words,

        "source_count":
            len(source_breakdown),

        "search_count":
            len(queries),

        "sentences_checked":
            sum(

                1

                for sentence
                in sentences

                if len(
                    tokenize(
                        sentence
                    )
                )
                >=
                MIN_SENTENCE_WORDS

            ),

        "matches":
            raw_matches,

        "sources":
            source_breakdown

    }


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    result = None


    if request.method == "POST":

        submitted_text = request.form.get(
            "text",
            ""
        ).strip()


        if not submitted_text:

            flash(
                "Please paste text into the text box."
            )


            return redirect(
                url_for("index")
            )


        try:

            result = analyze_text(
                submitted_text
            )


        except RuntimeError as error:

            print(
                "Configuration/API error:",
                error
            )


            flash(
                str(error)
            )


        except requests.RequestException as error:

            print(
                "Network error:",
                error
            )


            flash(
                "A web-search network error occurred. "
                "Please try again."
            )


        except Exception as error:

            print(
                "Unexpected error:",
                repr(error)
            )


            flash(
                "Analysis failed. Check the terminal "
                "for the exact error."
            )


    return render_template(
        "index.html",
        result=result
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    ) 