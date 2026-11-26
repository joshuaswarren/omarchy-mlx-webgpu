import { fetchCachedText } from "./cache";
import type { Qwen3Config } from "./config";

/**
 * Zero-dependency byte-level BPE tokenizer for Qwen3 (vocab.json + merges.txt + added_tokens.json).
 * Implements standard GPT-2/Qwen2.5/Qwen3 byte-to-unicode mapping, BPE rank merging,
 * special-token splitting, and incremental UTF-8 streaming decode.
 */

function buildByteToUnicode(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Array<string>(256);
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteToUnicode();

// Qwen2 / Qwen3 pre-tokenization pattern (ECMAScript Unicode property escapes)
const PRETOKENIZE_RE =
  /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;

export class Qwen3Tokenizer {
  private readonly vocab: Map<string, number>;
  private readonly idToToken: Map<number, string>;
  private readonly mergeRanks: Map<string, number>;
  private readonly specialTokens: Map<string, number>;
  private readonly specialRegex: RegExp | null;
  private readonly bpeCache = new Map<string, string[]>();
  private readonly utf8Encoder = new TextEncoder();
  private readonly utf8Decoder = new TextDecoder("utf-8", { fatal: false });

  constructor(
    vocabObj: Record<string, number>,
    mergesText: string,
    addedTokens: Record<string, number>,
  ) {
    this.vocab = new Map<string, number>();
    this.idToToken = new Map<number, string>();
    for (const [tok, id] of Object.entries(vocabObj)) {
      this.vocab.set(tok, id);
      this.idToToken.set(id, tok);
    }

    this.specialTokens = new Map<string, number>();
    for (const [tok, id] of Object.entries(addedTokens)) {
      this.specialTokens.set(tok, id);
      this.vocab.set(tok, id);
      this.idToToken.set(id, tok);
    }

    this.mergeRanks = new Map<string, number>();
    const lines = mergesText.split(/\r?\n/);
    let rank = 0;
    for (const line of lines) {
      if (!line || line.startsWith("#")) continue;
      this.mergeRanks.set(line, rank++);
    }

    const specials = [...this.specialTokens.keys()].sort((a, b) => b.length - a.length);
    if (specials.length > 0) {
      const escaped = specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      this.specialRegex = new RegExp(`(${escaped.join("|")})`, "g");
    } else {
      this.specialRegex = null;
    }
  }

  static async load(config: Qwen3Config): Promise<Qwen3Tokenizer> {
    const [vocabText, mergesText, addedText] = await Promise.all([
      fetchCachedText(`${config.baseUrl}/vocab.json`, "vocab.json"),
      fetchCachedText(`${config.baseUrl}/merges.txt`, "merges.txt"),
      fetchCachedText(`${config.baseUrl}/added_tokens.json`, "added_tokens.json"),
    ]);
    return new Qwen3Tokenizer(
      JSON.parse(vocabText) as Record<string, number>,
      mergesText,
      JSON.parse(addedText) as Record<string, number>,
    );
  }

  private bpe(token: string): string[] {
    const cached = this.bpeCache.get(token);
    if (cached) return cached;

    let word = Array.from(token);
    if (word.length <= 1) {
      this.bpeCache.set(token, word);
      return word;
    }

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.mergeRanks.get(`${word[i]} ${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;

      const first = word[bestIdx];
      const second = word[bestIdx + 1];
      const merged: string[] = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          merged.push(first + second);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
    }

    if (this.bpeCache.size < 20000) {
      this.bpeCache.set(token, word);
    }
    return word;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    const parts = this.specialRegex ? text.split(this.specialRegex) : [text];
    for (const part of parts) {
      if (!part) continue;
      const specialId = this.specialTokens.get(part);
      if (specialId !== undefined) {
        ids.push(specialId);
        continue;
      }
      const matches = part.match(PRETOKENIZE_RE);
      if (!matches) continue;
      for (const piece of matches) {
        const bytes = this.utf8Encoder.encode(piece);
        let mapped = "";
        for (let i = 0; i < bytes.length; i++) {
          mapped += BYTE_TO_CHAR[bytes[i]];
        }
        for (const sub of this.bpe(mapped)) {
          const id = this.vocab.get(sub);
          if (id !== undefined) ids.push(id);
        }
      }
    }
    return ids;
  }

  decode(ids: number[], skipSpecial = true): string {
    const bytes: number[] = [];
    let out = "";
    const flushBytes = () => {
      if (bytes.length > 0) {
        out += this.utf8Decoder.decode(new Uint8Array(bytes));
        bytes.length = 0;
      }
    };
    for (const id of ids) {
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      if (this.specialTokens.has(tok)) {
        if (!skipSpecial) {
          flushBytes();
          out += tok;
        }
        continue;
      }
      for (const ch of tok) {
        const b = CHAR_TO_BYTE.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flushBytes();
    return out;
  }

  /** Formats a Qwen3 chat prompt with direct (no-think) assistant prefill for crisp streaming. */
  formatChatPrompt(userPrompt: string, systemPrompt = "You are a concise, clear technical assistant."): string {
    return (
      `<|im_start|>system\n${systemPrompt}<|im_end|>\n` +
      `<|im_start|>user\n${userPrompt}<|im_end|>\n` +
      `<|im_start|>assistant\n<think>\n\n</think>\n\n`
    );
  }
}
