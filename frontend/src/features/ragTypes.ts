export type Chapter = {
  chapter_id: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
  content: string;
  char_count: number;
};

export type Textbook = {
  textbook_id: string;
  filename: string;
  title: string;
  total_pages: number | null;
  total_chars: number;
  chapters: Chapter[];
};

export type RagIndexResponse = {
  index_id: string;
  textbook_count: number;
  chapter_count: number;
  chunk_count: number;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string | null;
  embedding_count: number;
  vector_store_type: string;
  chunking_strategy: string;
};

export type RagIndexJobStartResponse = {
  job_id: string;
  status: string;
  index_id: string;
  estimated_chunk_count: number;
};

export type RagIndexStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  index_id: string;
  textbook_count: number;
  chapter_count: number;
  chunk_count: number;
  embedding_count: number;
  estimated_chunk_count: number;
  vector_store_type: string;
  error: string | null;
  result: RagIndexResponse | null;
};

export type RagChunkMetadata = {
  textbook_id: string;
  textbook_title: string;
  filename: string;
  chapter_id: string;
  chapter_title: string;
  page_start: number | null;
  page_end: number | null;
  chunk_index: number;
  char_start: number;
  char_end: number;
  retrieval_model: string;
  embedding_model: string | null;
  chunk_size: number;
  chunk_overlap: number;
};

export type RagCitation = {
  citation_id: string;
  chunk_id: string;
  textbook_id: string;
  textbook_title: string;
  filename: string;
  chapter_id: string;
  chapter_title: string;
  page_start: number | null;
  page_end: number | null;
  score: number;
};

export type RagSourceChunk = {
  chunk_id: string;
  text: string;
  score: number;
  metadata: RagChunkMetadata;
};

export type RagQueryResponse = {
  answer: string;
  citations: RagCitation[];
  source_chunks: RagSourceChunk[];
  retrieval: {
    index_id: string;
    retrieval_mode: "hybrid" | "keyword" | "char_ngram" | "vector";
    retrieval_model: string;
    embedding_model: string | null;
    embedding_count: number;
    vector_store_type: string;
    chunking_strategy: string;
    top_k: number;
    min_score: number;
    matched_chunks: number;
  };
};
