/**
 * Token counting utilities using tiktoken
 * Used for accurate token estimation for prompt caching and chunking
 */

import { encoding_for_model, TiktokenModel } from "tiktoken";

// Claude uses cl100k_base encoding (same as GPT-4)
const ENCODING_NAME = "cl100k_base";

let encoder: ReturnType<typeof encoding_for_model> | null = null;

/**
 * Get or create the tiktoken encoder
 */
function getEncoder() {
  if (!encoder) {
    // Use gpt-4 as proxy since Claude uses similar tokenization
    encoder = encoding_for_model("gpt-4" as TiktokenModel);
  }
  return encoder;
}

/**
 * Count tokens in a string
 */
export function countTokens(text: string): number {
  if (!text) return 0;
  const enc = getEncoder();
  const tokens = enc.encode(text);
  return tokens.length;
}

/**
 * Truncate text to a maximum number of tokens
 */
export function truncateToTokens(text: string, maxTokens: number): string {
  if (!text) return "";
  const enc = getEncoder();
  const tokens = enc.encode(text);

  if (tokens.length <= maxTokens) {
    return text;
  }

  const truncatedTokens = tokens.slice(0, maxTokens);
  return enc.decode(truncatedTokens);
}

/**
 * Split text into chunks of approximately equal token count
 */
export function splitByTokens(
  text: string,
  maxTokensPerChunk: number,
  overlapTokens: number = 0
): string[] {
  if (!text) return [];

  const enc = getEncoder();
  const tokens = enc.encode(text);
  const chunks: string[] = [];

  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + maxTokensPerChunk, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    chunks.push(enc.decode(chunkTokens));

    // Move start forward, accounting for overlap
    start = end - overlapTokens;
    if (start >= tokens.length || end === tokens.length) break;
  }

  return chunks;
}

/**
 * Estimate tokens for a message structure (for prompt caching)
 */
export function estimateMessageTokens(messages: Array<{ role: string; content: string }>): number {
  let total = 0;
  for (const msg of messages) {
    // Add overhead for message structure (~4 tokens per message)
    total += 4;
    total += countTokens(msg.content);
  }
  // Add overhead for the overall structure
  total += 3;
  return total;
}

/**
 * Check if content meets minimum token threshold for caching
 * Claude Sonnet: 1024 tokens minimum
 * Claude Opus: 4096 tokens minimum
 */
export function meetsMinCacheThreshold(
  tokenCount: number,
  model: "sonnet" | "opus" | "haiku" = "sonnet"
): boolean {
  const thresholds = {
    sonnet: 1024,
    opus: 4096,
    haiku: 2048,
  };
  return tokenCount >= thresholds[model];
}

/**
 * Get token statistics for content
 */
export function getTokenStats(text: string): {
  totalTokens: number;
  characters: number;
  tokensPerChar: number;
  estimatedCost: {
    input: number;
    cachedRead: number;
    cachedWrite: number;
  };
} {
  const totalTokens = countTokens(text);
  const characters = text.length;

  // Pricing per million tokens (Claude Sonnet 4.5)
  const inputPricePerMillion = 3;
  const cacheReadPricePerMillion = 0.3; // 10% of input
  const cacheWritePricePerMillion = 3.75; // 125% of input

  return {
    totalTokens,
    characters,
    tokensPerChar: characters > 0 ? totalTokens / characters : 0,
    estimatedCost: {
      input: (totalTokens / 1_000_000) * inputPricePerMillion,
      cachedRead: (totalTokens / 1_000_000) * cacheReadPricePerMillion,
      cachedWrite: (totalTokens / 1_000_000) * cacheWritePricePerMillion,
    },
  };
}

/**
 * Clean up the encoder when done
 */
export function freeEncoder(): void {
  if (encoder) {
    encoder.free();
    encoder = null;
  }
}
