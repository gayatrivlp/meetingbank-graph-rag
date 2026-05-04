# Meetingbank Graph RAG
MeetingBank Council Intelligence — Graph RAG
A Graph RAG system for querying 1,250 U.S. city council meetings across 6 cities, built on Neo4j and deployed as a live Streamlit application. The system combines vector similarity search on transcript chunks with graph traversal for structured metadata filtering — enabling questions that flat vector search alone cannot answer reliably.

**Live Demo**->  https://meetingbank-graph-rag.streamlit.app/

## Why Graph RAG?
Flat vector search finds semantically similar chunks of text — it answers "what does this text look like?" But it has no awareness of structure: how documents relate to each other, what type of entity a chunk belongs to, or how to traverse from one entity to another.
Metadata filtering doesn't solve this. You can attach metadata to vectors (city, item type, date) and filter post-retrieval — tools like Pinecone support this. But metadata filtering is applied after the vector search, on a flat list of results. It can narrow results, but it cannot traverse relationships. The question "find all items discussed in the same meeting as this housing ordinance" cannot be expressed as a metadata filter, because the relationship between items is structural, not a property stored on any single chunk.
Graph RAG adds a knowledge graph layer where entities are nodes and their relationships are first-class citizens. This enables things flat vector search fundamentally cannot do:
- *Relationship traversal*. A chunk is connected to the item it came from, which is connected to the meeting it belongs to, which is connected to the city that hosted it. Graph RAG can traverse this path in a single query — returning not just the chunk but the full structured context of everything it's connected to.
- *Multi-hop reasoning*. Starting from a semantically retrieved chunk, a graph query can hop to sibling nodes — other agenda items in the same meeting, other meetings in the same city, or all items of a given type across the entire dataset. This kind of reasoning requires traversing relationships, which a vector index has no concept of.
- *Enrichment at query time*. Each retrieved result arrives pre-enriched with structured metadata from the graph — city, meeting date, item type, summary, source links — resolved through traversal in a single Cypher query. This is structurally cleaner than a separate metadata lookup against a secondary store.

**What this project implements?**\
This system uses two-stage Graph RAG: Stage 1 runs vector similarity search on *TranscriptChunk* nodes, Stage 2 traverses the graph from those chunks to retrieve the full structured context of each matched item (city, meeting, item type, summary, links) and applies city and item type filters during traversal. The LLM then reasons over the enriched context to synthesize an answer.

## Dataset
**MeetingBank** (Hu et al., ACL 2023) — a benchmark dataset of 1,366 city council meetings from 6 U.S. municipalities:

The MeetingBank dataset is a natural fit for this architecture. It has a well-defined entity hierarchy (City → Meeting → Item → TranscriptChunk), structured metadata (item type, duration, links), and domain-specific reference summaries that made evaluation straightforward without needing external labels.
