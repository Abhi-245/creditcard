import os
import re
import hashlib
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import cohere
# ----------------------------
# Load environment
# ----------------------------
load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env")

from sentence_transformers import CrossEncoder

# Load locally if already saved, else download
RERANKER_MODEL_PATH = "./models/ms-marco-MiniLM-L12-v2"
reranker = CrossEncoder(RERANKER_MODEL_PATH)

co = cohere.Client("CnIM7F98SxdbRxAhth2MIBBthcDlFWPITfg4hUa5")

# ----------------------------
# OpenAI + Chroma setup
# ----------------------------
client = OpenAI(api_key=OPENAI_KEY)
chroma_client = chromadb.PersistentClient(path="./docchroma_store")

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"
)

COLLECTION_NAME = "credit_cards"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)
# collection = chroma_client.get_collection("credit_cards")
# print(collection.count())   # number of embeddings

# ----------------------------
# Utilities
# ----------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(text, chunk_size=800, overlap=80):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(words)), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks


# Put this helper near the top of your file (imports already present)
def build_chroma_where(filters: dict | None):
    """
    Convert a plain filters dict into a Chroma 'where' clause:
      - If filters is None/empty -> return None
      - If filters contains a single condition -> return that condition dict (e.g. {"card_name": {"$in": [...]}})
      - If multiple conditions -> return {"$and": [cond1, cond2, ...]}
    Non-dict values are converted to {"$eq": value}.
    """
    if not filters:
        return None

    clauses = []
    for key, val in filters.items():
        # If val already looks like a Chroma operator dict (e.g. {"$in": [...]}) keep it
        if isinstance(val, dict):
            clauses.append({key: val})
        else:
            # convert simple value into an equality clause
            clauses.append({key: {"$eq": val}})

    if len(clauses) == 1:
        # return the single condition dict directly
        return clauses[0]
    # multiple conditions -> use $and
    return {"$and": clauses}



import difflib
import itertools

def get_all_cards_and_aliases():
    """
    Read metadata from Chroma and return a dict:
      { "Card Name": set(aliases...) }
    Robust to Chroma returning nested metadata lists.
    """
    all_cards = {}
    try:
        results = collection.get(include=["metadatas"], where={"type": "text"}, limit=10000)
        metadatas = results.get("metadatas") or []

        # if Chroma returns nested list(s) like [[meta1, meta2, ...]]
        if len(metadatas) and isinstance(metadatas[0], list):
            metas_iter = [m for sub in metadatas for m in sub]
        else:
            metas_iter = metadatas

        for meta in metas_iter:
            if not meta:
                continue
            card = meta.get("card_name")
            if not card:
                continue
            card_key = card  # keep canonical card name as key
            if card_key not in all_cards:
                all_cards[card_key] = set()

            # Add stored aliases (if list or single string)
            aliases = meta.get("aliases")
            if aliases:
                if isinstance(aliases, (list, set, tuple)):
                    all_cards[card_key].update([a.lower() for a in aliases if a])
                elif isinstance(aliases, str):
                    all_cards[card_key].add(aliases.lower())

            # Always add canonical lower-cased card name and a cleaned form
            all_cards[card_key].add(card.lower())
            # also add card without 'credit card'
            cleaned = card.lower().replace(" credit card", "").strip()
            all_cards[card_key].add(cleaned)

    except Exception as e:
        print("⚠️ Could not fetch cards from collection:", e)
    return all_cards


def generate_aliases(card_name: str):
    """
    Create many helpful aliases/permutations for the card_name.
    E.g. "Flipkart Axis Bank Credit Card" -> "flipkart axis bank", "axis flipkart", "flipkart axis", ...
    """
    card_lower = card_name.lower().strip()
    aliases = set([card_lower])

    # Remove "credit card"
    no_cc = card_lower.replace(" credit card", "").strip()
    aliases.add(no_cc)

    # Remove common words like 'bank' sometimes
    no_bank = no_cc.replace(" bank", "").strip()
    aliases.add(no_bank)

    parts = [p for p in re.findall(r"\w+", no_cc)]
    if len(parts) > 1:
        # Add first+last, last+first
        aliases.add(parts[0] + " " + parts[-1])
        aliases.add(parts[-1] + " " + parts[0])

    # Add permutations up to length 3 (safety net)
    for perm in itertools.permutations(parts, min(len(parts), 3)):
        aliases.add(" ".join(perm))

    # Short forms: first two words, last two words
    if len(parts) >= 2:
        aliases.add(" ".join(parts[:2]))
        aliases.add(" ".join(parts[-2:]))

    # Normalize and return
    return [a.strip() for a in aliases if a]


import difflib
import math

# def detect_cards_in_query(query, known_cards):
#     """
#     More conservative card detection:
#       - ignores generic words like 'credit', 'card', 'bank' when matching
#       - requires >=2 match-points (direct token matches score 2, fuzzy matches score 1)
#       - falls back to all-cards only when no specific card is detected
#     """
    
#     print(f"known cards:{known_cards}")
#     STOP_TOKENS = {"credit", "card", "cards", "bank", "creditcard", "carding"}
#     q_lower = query.lower()
#     q_words = set(re.findall(r"\w+", q_lower))

#     all_cards = get_all_cards_and_aliases()
#     if not all_cards:
#         return []

#     detected = []

#     for card, aliases in all_cards.items():
#         # make sure aliases is iterable
#         for alias in aliases:
#             if not alias:
#                 continue
#             alias = alias.lower().strip()
#             alias_words = set(re.findall(r"\w+", alias))

#             # drop generic tokens from alias word set
#             alias_nonstop = [w for w in alias_words if w not in STOP_TOKENS]
#             # if alias is only generic tokens (e.g., "credit card"), skip it
#             if not alias_nonstop:
#                 continue

#             # fast check: exact substring
#             if alias in q_lower:
#                 detected.append(card)
#                 break

#             # scoring: direct token match = 2 points, fuzzy token match = 1 point
#             score = 0
#             for aw in alias_nonstop:
#                 if aw in q_words:
#                     score += 2
#                 else:
#                     # only try fuzzy for reasonably long tokens (avoid short noise)
#                     if len(aw) >= 4:
#                         close = difflib.get_close_matches(aw, list(q_words), n=1, cutoff=0.85)
#                         if close:
#                             score += 1

#             # require at least 2 points (so one direct token match OR two fuzzy/partial matches)
#             if score >= 2:
#                 detected.append(card)
#                 break

#     detected = list(set(detected))

#     # If none detected -> fallback to "compare all" only when query really asks generic advice
#     # if not detected:
#     #     triggers = [
#     #         "which card", "which cards", "suggest", "suggestion", "suggestions",
#     #         "recommend", "recommendation", "compare", "comparison", "when to use",
#     #         "usage", "use", "best card", "advice", "tips"
#     #     ]
#     #     if any(t in q_lower for t in triggers):
#     #         detected = list(all_cards.keys())
            
#     if not detected:
#         triggers = [
#             "which card", "which cards", "suggest", "suggestion", "suggestions",
#             "recommend", "recommendation", "compare", "comparison", "when to use",
#             "usage", "use", "best card", "advice", "tips"
#             ]
#         if any(t in q_lower for t in triggers):
#             # only fallback to all cards if query really generic
#             detected = list(all_cards.keys())
        
#         else:
#             # small fuzzy check for trigger tokens (handles typos like "suggesstions")
#             q_tokens = re.findall(r"\w+", q_lower)
#             for token in q_tokens:
#                 for t in triggers:
#                     for t_word in t.split():
#                         if difflib.get_close_matches(token, [t_word], cutoff=0.8):
#                             detected = list(all_cards.keys())
#                             break
#                     if detected:
#                         break
#                 if detected:
#                     break

#         # last resort: if query explicitly contains both 'credit' and 'card' tokens treat as generic
#         if not detected:
#             if "credit" in q_words and ("card" in q_words or "cards" in q_words):
#                 detected = list(all_cards.keys())

#     return list(set(detected))

# def detect_cards_in_query_strict(query):
#     """
#     Strict card detection:
#       - Ignores generic words like 'credit', 'card', 'bank'.
#       - Requires >=2 significant token matches to detect a card.
#       - Uses exact match first, then fuzzy match for long tokens.
#       - Avoids false positives like SBI or HSBC unless explicitly mentioned.
#     """
#     STOP_TOKENS = {"credit", "card", "cards", "bank", "creditcard", "carding"}
#     q_lower = query.lower()
#     q_words = set(re.findall(r"\w+", q_lower))
#     q_words_nonstop = {w for w in q_words if w not in STOP_TOKENS}

#     all_cards = get_all_cards_and_aliases()
#     if not all_cards:
#         return []

#     detected = []

#     for card, aliases in all_cards.items():
#         card_detected = False
#         for alias in aliases:
#             if not alias:
#                 continue
#             alias_words = set(re.findall(r"\w+", alias.lower()))
#             alias_nonstop = {w for w in alias_words if w not in STOP_TOKENS}

#             if not alias_nonstop:
#                 continue

#             # Exact substring match first
#             if alias in q_lower:
#                 card_detected = True
#                 break

#             # Token-based scoring: 2 points exact, 1 point fuzzy
#             score = 0
#             for aw in alias_nonstop:
#                 if aw in q_words_nonstop:
#                     score += 2
#                 elif len(aw) >= 4:  # fuzzy only for longer tokens
#                     close = difflib.get_close_matches(aw, q_words_nonstop, n=1, cutoff=0.85)
#                     if close:
#                         score += 1

#             # Require at least 3 points to consider a card detected
#             if score >= 3:
#                 card_detected = True
#                 break

#         if card_detected:
#             detected.append(card)

#     return list(set(detected))


def detect_cards_in_query_strict(query):
    """
    Improved strict detection:
      - Keeps STOP_TOKENS filtering
      - Adds fallback for 'all cards' queries
      - Loosens fuzzy cutoff slightly
      - Allows detection with 1 strong fuzzy match + bank keyword
    """
    STOP_TOKENS = {"credit", "card", "cards", "bank", "creditcard", "carding"}
    q_lower = query.lower()
    q_words = set(re.findall(r"\w+", q_lower))
    q_words_nonstop = {w for w in q_words if w not in STOP_TOKENS}

    all_cards = get_all_cards_and_aliases()
    if not all_cards:
        return []

    detected = []

    for card, aliases in all_cards.items():
        card_detected = False
        for alias in aliases:
            if not alias:
                continue
            alias_words = set(re.findall(r"\w+", alias.lower()))
            alias_nonstop = {w for w in alias_words if w not in STOP_TOKENS}
            if not alias_nonstop:
                continue

            # Exact substring match
            if alias in q_lower:
                card_detected = True
                break

            # Token-based scoring
            score = 0
            for aw in alias_nonstop:
                if aw in q_words_nonstop:
                    score += 2
                elif len(aw) >= 4:
                    close = difflib.get_close_matches(aw, q_words_nonstop, n=1, cutoff=0.80)  # relaxed cutoff
                    if close:
                        score += 1

            # Allow detection if strong partial match + bank keyword present
            if score >= 3 or (score >= 1 and any(b in q_words_nonstop for b in ["hdfc", "sbi", "axis", "hsbc"])):
                card_detected = True
                break

        if card_detected:
            detected.append(card)

    # Fallback: query asks for "all cards"
    if not detected and re.search(r"\ball cards\b|\bcompare cards\b", q_lower):
        detected = list(all_cards.keys())

    return list(set(detected))


def detect_category(query):
    category_map = {
        "groceries": ["grocery", "supermarket", "food", "mart"],
        "fuel": ["fuel", "petrol", "diesel", "gas"],
        "travel": ["flight", "hotel", "rail", "travel"],
        "dining": ["restaurant", "dining", "cafe", "eatery"]
    }
    q_lower = query.lower()
    for cat, keywords in category_map.items():
        for kw in keywords:
            if kw in q_lower:
                return cat
    return None

def extract_merchants(txt: str):
    merchants = []
    for line in txt.splitlines():
        line = line.strip()
        if re.match(r"^\d+\s", line):
            merchants.append(line)
    return merchants

# ----------------------------
# Indexing function
# ----------------------------
# def index_card_folder(bank_name: str, card_name: str, folder_path: str,
#                       chunk_size=800, overlap=80, allow_reindex = True, batch_size=100):
#     folder_path = os.path.abspath(folder_path)
#     added = 0

#     for fn in os.listdir(folder_path):
#         full = os.path.join(folder_path, fn)
#         if not os.path.isfile(full):
#             continue
#         # if not (fn.lower().endswith("combined.md") or fn.endswith("list-of-stores.pdf.txt")):
#         #     continue
#         if not (fn.lower().endswith(".md") or fn.lower().endswith("list-of-stores.pdf.txt")):
#             continue
#         print(f"-> Processing {full}")
#         with open(full, "r", encoding="utf-8", errors="ignore") as f:
#             txt = f.read()

#         if not txt.strip():
#             print("  (no text found, skip)")
#             continue

#         # Index text chunks
#         chunks = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
#         docs, ids, metadatas = [], [], []
#         for i, chunk in enumerate(chunks):
#             chunk_id = sha1(fn + "_chunk_" + str(i))
#             if not allow_reindex:
#                 try:
#                     existing = collection.get(ids=[chunk_id])
#                     if existing and existing.get("ids"):
#                         continue
#                 except Exception:
#                     pass

#             docs.append(chunk)
#             ids.append(chunk_id)
#             metadatas.append({
#                 "source": fn,
#                 "bank": bank_name,
#                 "card_name": card_name,
#                 "aliases": ", ".join(generate_aliases(card_name)),
#                 #  "aliases": generate_aliases(card_name),
#                 # "aliases" : [a for a in generate_aliases(card_name) if not any(w in STOP_TOKENS for w in a.split())],

#                 "type": "text",
#                 "chunk_index": i
#             })
            
#             if len(docs) >= batch_size:
#                 collection.add(documents=docs, ids=ids, metadatas=metadatas)
#                 added += len(docs)
#                 docs, ids, metadatas = [], [], []

#         if docs:
#             collection.add(documents=docs, ids=ids, metadatas=metadatas)
#             added += len(docs)

#         # Index merchants separately
#         merchants = extract_merchants(txt)
#         if merchants:
#             print(f"  Found {len(merchants)} merchant rows, indexing individually...")
#             docs, ids, metadatas = [], [], []
#             for i, merchant in enumerate(merchants):
#                 chunk_id = sha1(fn + "_merchant_" + str(i))
#                 if not allow_reindex:
#                     try:
#                         existing = collection.get(ids=[chunk_id])
#                         if existing and existing.get("ids"):
#                             continue
#                     except Exception:
#                         pass
#                 docs.append(merchant)
#                 ids.append(chunk_id)
#                 metadatas.append({
#                     "source": fn,
#                     "bank": bank_name,
#                     "card_name": card_name,
#                     "type": "merchant",
#                     "row": i
#                 })

#                 if len(docs) >= batch_size:
#                     collection.add(documents=docs, ids=ids, metadatas=metadatas)
#                     added += len(docs)
#                     docs, ids, metadatas = [], [], []
                    
          
#             if docs:
#                 collection.add(documents=docs, ids=ids, metadatas=metadatas)
#                 added += len(docs)
                
               
                    
                    
#         print(f"  Indexed {added} chunks from {fn}")

#     print(f"\n✅ Done indexing {card_name} ({bank_name}). Added {added} chunks to collection '{COLLECTION_NAME}'")

def index_card_folder(bank_name: str, card_name: str, folder_path: str,
                      chunk_size=800, overlap=80, allow_reindex=False, batch_size=100):
    folder_path = os.path.abspath(folder_path)
    added = 0

    for fn in os.listdir(folder_path):
        full = os.path.join(folder_path, fn)
        if not os.path.isfile(full):
            continue
        if not (fn.lower().endswith(".md") or fn.lower().endswith("list-of-stores.pdf.txt")):
            continue

        print(f"-> Processing {full}")
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        if not txt.strip():
            print("  (no text found, skip)")
            continue

        file_added = 0

        # ----------- Text chunks -----------
        chunks = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
        print(f"  Generated {len(chunks)} text chunks")
        docs, ids, metadatas = [], [], []

        for i, chunk in enumerate(chunks):
            chunk_id = sha1(full + "_chunk_" + str(i))  # include full path for uniqueness
            if not allow_reindex:
                try:
                    existing = collection.get(ids=[chunk_id])
                    if existing and existing.get("ids"):
                        continue
                except Exception:
                    pass

            docs.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "source": fn,
                "bank": bank_name,
                "card_name": card_name,
                "aliases": ", ".join(generate_aliases(card_name)),
                "type": "text",
                "chunk_index": i
            })

            if len(docs) >= batch_size:
                print(f"  Adding batch of {len(docs)} text docs")
                collection.add(documents=docs, ids=ids, metadatas=metadatas)
                file_added += len(docs)
                docs, ids, metadatas = [], [], []

        # Add remaining text chunks
        if docs:
            print(f"  Adding remaining {len(docs)} text docs")
            collection.add(documents=docs, ids=ids, metadatas=metadatas)
            file_added += len(docs)

        # ----------- Merchant rows -----------
        merchants = extract_merchants(txt)
        if merchants:
            print(f"  Found {len(merchants)} merchant rows, indexing individually...")
            docs, ids, metadatas = [], [], []

            for i, merchant in enumerate(merchants):
                chunk_id = sha1(full + "_merchant_" + str(i))
                if not allow_reindex:
                    try:
                        existing = collection.get(ids=[chunk_id])
                        if existing and existing.get("ids"):
                            continue
                    except Exception:
                        pass

                docs.append(merchant)
                ids.append(chunk_id)
                metadatas.append({
                    "source": fn,
                    "bank": bank_name,
                    "card_name": card_name,
                    "type": "merchant",
                    "row": i
                })

                if len(docs) >= batch_size:
                    print(f"  Adding batch of {len(docs)} merchant docs")
                    collection.add(documents=docs, ids=ids, metadatas=metadatas)
                    file_added += len(docs)
                    docs, ids, metadatas = [], [], []

            if docs:
                print(f"  Adding remaining {len(docs)} merchant docs")
                collection.add(documents=docs, ids=ids, metadatas=metadatas)
                file_added += len(docs)

        added += file_added
        print(f"  Indexed {file_added} chunks from {fn}")

    print(f"\n✅ Done indexing {card_name} ({bank_name}). Added {added} chunks to collection '{COLLECTION_NAME}'")


# ----------------------------
# Query / Retrieval
# ----------------------------

from sentence_transformers import CrossEncoder

def debug_collection(collection):
    total = collection.count()
    print(f"DEBUG: collection has {total} documents")

    if total == 0:
        print("⚠️ No documents found!")
        return

    # res = collection.get(include=["metadatas"], limit=total)

    # Count chunks per card
    # card_counts = {}
    # for meta in res["metadatas"]:
    #     card_name = meta["card_name"]
    #     card_counts[card_name] = card_counts.get(card_name, 0) + 1

    # print("Cards in collection with number of chunks:")
    # for card, count in card_counts.items():
    #     print(f" - {card}: {count} chunks")


    res = collection.get(include=["metadatas"], limit=collection.count())
    metas = res["metadatas"]

    # Flatten nested lists if needed
    flat_metas = []
    for m in metas:
        if isinstance(m, list):
            flat_metas.extend(m)
        else:
            flat_metas.append(m)

    # Count chunks per card
    from collections import Counter
    card_counter = Counter(meta["card_name"] for meta in flat_metas if "card_name" in meta)

    print("Cards in collection with number of chunks:")
    for card, count in card_counter.items():
        print(f" - {card}: {count} chunks")



def answer_query(query, n_results=40, use_cohere=True):
    known_cards = ["SBI Cashback Card", "Flipkart Axis Bank Credit Card", "HSBC LIVE+", "HDFC Diners Privilege Credit Card"]
    matched_cards = detect_cards_in_query_strict(query)
    print(f"matched_cards:{matched_cards}")
    category = detect_category(query)

    # ----------------------
    # Build filters for Chroma
    # ----------------------
    filters = {}
    if matched_cards:
        filters["card_name"] = {"$in": matched_cards}

    if "merchant" in query.lower() or re.search(r"in\s+[a-zA-Z]+", query):
        filters["type"] = "merchant"

    chroma_filters = build_chroma_where(filters)
    print("DEBUG filters:", chroma_filters)

    # ----------------------
    # Query Chroma
    # ----------------------
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=chroma_filters
        )
    except ValueError as e:
        print("ERROR: Chroma query validation failed. where=", chroma_filters)
        raise

    retrieved_docs = results["documents"][0] if isinstance(results["documents"], list) else results["documents"]
    retrieved_metas = results["metadatas"][0] if isinstance(results["metadatas"], list) else results["metadatas"]

    # ==========================================================
    # RERANKER LOGIC
    # ==========================================================
    if use_cohere and retrieved_docs:
        # ---- Cohere Reranker ----
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=retrieved_docs,
            top_n=5
        )

        ranked = []
        for r in rerank_response.results:
            idx = r.index
            score = r.relevance_score
            if idx is not None and idx < len(retrieved_docs):
                doc = retrieved_docs[idx]
                meta = retrieved_metas[idx] if idx < len(retrieved_metas) else {}
                ranked.append((doc, meta, score))

        if ranked:
            retrieved_docs, retrieved_metas, rerank_scores = zip(*ranked)
            retrieved_docs, retrieved_metas, rerank_scores = (
                list(retrieved_docs),
                list(retrieved_metas),
                list(rerank_scores),
            )
        else:
            retrieved_docs, retrieved_metas, rerank_scores = [], [], []

        print("\n--- Top 5 snippets after Cohere reranker ---")
        for i, (doc, meta, score) in enumerate(zip(retrieved_docs[:5], retrieved_metas[:5], rerank_scores[:5])):
            snippet = (doc[:400] + "...") if len(doc) > 400 else doc
            print(f"[{i}] score={score:.4f} card={meta.get('card_name','unknown')} "
                  f"bank={meta.get('bank','unknown')} snippet={snippet!r}")

    elif not use_cohere and retrieved_docs:
        # ---- MiniLM CrossEncoder Reranker ----
        pairs = [(query, doc) for doc in retrieved_docs if doc]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(retrieved_docs, retrieved_metas, scores),
            key=lambda x: x[2],
            reverse=True
        )

        retrieved_docs, retrieved_metas, rerank_scores = zip(*ranked)
        retrieved_docs, retrieved_metas, rerank_scores = (
            list(retrieved_docs),
            list(retrieved_metas),
            list(rerank_scores),
        )

        print("\n--- Top 5 snippets after MiniLM reranker ---")
        for i, (doc, meta, score) in enumerate(zip(retrieved_docs[:5], retrieved_metas[:5], rerank_scores[:5])):
            snippet = (doc[:400] + "...") if len(doc) > 400 else doc
            print(f"[{i}] score={score:.4f} card={meta.get('card_name','unknown')} "
                  f"bank={meta.get('bank','unknown')} snippet={snippet!r}")

    else:
        retrieved_docs, retrieved_metas = [], []

    # ----------------------
    # Category filtering after rerank
    # ----------------------
    if category:
        filtered = [(doc, meta) for doc, meta in zip(retrieved_docs, retrieved_metas) if category in doc.lower()]
        if filtered:
            retrieved_docs, retrieved_metas = zip(*filtered)
            retrieved_docs, retrieved_metas = list(retrieved_docs), list(retrieved_metas)

    # ----------------------
    # Retrieved snippets printing
    # ----------------------
    # print("\n--- Retrieved snippets (final after rerank + filters) ---")
    # for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
    #     if not doc:
    #         continue
    #     snippet = (doc[:400] + "...") if len(doc) > 400 else doc
    #     print(f"[{i}] source={meta.get('source','unknown')} "
    #           f"bank={meta.get('bank','unknown')} card={meta.get('card_name','unknown')} snippet={snippet!r}")

    # ----------------------
    # Handle empty results
    # ----------------------
    if not retrieved_docs:
        return "Sorry, I can’t help you with this!!"

    # ----------------------
    # Context passed to LLM
    # ----------------------
    top_k = 5
    top_context = "\n\n".join(retrieved_docs[:top_k])

    # ----------------------
    # Prompt
    # ----------------------
    prompt = f"""
You are a helpful assistant specializing in credit cards.

You are given:
- A user query
- Context snippets retrieved from the knowledge base (credit card T&Cs, features, merchant offers, fees, etc.)

Your rules:
1. **Ground answers only in the retrieved context which is in markdown format.** If something is not in the context, do not invent it.
2. **When responding to generic queries always go through all the context provided of all cards and compare them before answering.**
3. **Always mention card names explicitly** (e.g., "SBI Cashback Card") when discussing benefits.
4. If multiple cards are relevant, **compare them side by side** (e.g., "Card A gives 5% on dining, Card B gives 3% on groceries").
5. If the query is generic ("best for dining", "which card to suggest"), summarize the retrieved context into a **clear recommendation**.
6. If context is sparse:
   - Say what you *can* infer from the snippets.
   - Politely note what is missing (instead of saying "Sorry, I can't help").
7. For category queries (groceries, dining, travel, fuel), focus only on relevant benefits from context.
8. Answer in a **clear, concise, and user-friendly tone** — short paragraphs or bullet points are preferred.

---
Context:
{top_context}

User Question:
{query}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Index cards
    
    index_card_folder("SBI", "SBI Cashback Card", r"SBI\SBI CASHBACK CREDITCARD", chunk_size=800, overlap=80,allow_reindex = False,batch_size=100)
    index_card_folder("Axis Bank", "Flipkart Axis Bank Credit Card", r"AXISBANK/FLIPKARTAXISBANK", chunk_size=800, overlap=80,allow_reindex = False,batch_size=100)
    index_card_folder("HSBC", "HSBC LIVE+", r"HSBC/HSBCLIVE+", chunk_size=800, overlap=80,allow_reindex = False,batch_size=100)
    index_card_folder("HDFC", "Diners Club Privilege Credit Card", r"HDFC/DinersClubPrivilegeCreditCard", chunk_size=800, overlap=80,allow_reindex = False,batch_size=100)
    
    
    print(debug_collection(collection))

    
    queries = [
        
        # "What are the Detailed Service Guides and Cardholder Agreements for Flipkart axis bank card?",
            "Which card shall i use to buy coffee, milk and other general things online, compare all cards and then recommend one card?",
             "Give me Details about HDFC Dinners privilege card.",
            #  "List the merchants in bangalore, i can use SBI Cashback credit card"
            ]

    for q in queries:
        print("\n=== QUERY:", q)
        ans = answer_query(q, n_results=40)
        print("\n--- ANSWER ---\n", ans)
