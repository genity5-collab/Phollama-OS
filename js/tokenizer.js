'use strict';
/**
 * Nova tokenizer — custom, character-level, with stable token IDs.
 *
 * Fixed IDs (stable forever, part of the tokenizer's definition):
 *   0 <pad>, 1 <unk>, 2 <bos>, 3 <eos>
 *   4-98    printable ASCII (char codes 32-126)
 *   99      newline
 *
 * Additional characters beyond the base alphabet (accented letters, emoji,
 * ...) get IDs 100+ in first-seen order while the vocabulary is built from
 * the training corpus. The vocabulary is serialized with the model, which
 * keeps every token ID stable across sessions.
 */
class NovaTokenizer {
  constructor() {
    // Fixed start: the four special tokens, then the printable ASCII
    // alphabet (char codes 32-126, IDs 4-98), then newline (ID 99).
    // These IDs are part of the tokenizer's definition and never change.
    this.itos = ['<pad>', '<unk>', '<bos>', '<eos>'];
    for (let c = 32; c <= 126; c++) this.itos.push(String.fromCharCode(c));
    this.itos.push('\n');
    this.stoi = null;
    this._rebuild();
  }

  static get PAD() { return 0; }
  static get UNK() { return 1; }
  static get BOS() { return 2; }
  static get EOS() { return 3; }

  _rebuild() {
    this.stoi = Object.create(null);
    for (let i = 0; i < this.itos.length; i++) this.stoi[this.itos[i]] = i;
  }

  /** Adds every new character of `text` to the vocabulary. */
  buildFromText(text) {
    for (const ch of text) this.addChar(ch);
  }

  addChar(ch) {
    if (this.stoi[ch] === undefined) {
      this.stoi[ch] = this.itos.length;
      this.itos.push(ch);
    }
  }

  vocabSize() { return this.itos.length; }

  /** text -> token IDs, always framed with <bos> ... <eos>. */
  encode(text) {
    const out = [NovaTokenizer.BOS];
    for (const ch of text) {
      out.push(this.stoi[ch] !== undefined ? this.stoi[ch] : NovaTokenizer.UNK);
    }
    out.push(NovaTokenizer.EOS);
    return out;
  }

  /** token IDs -> text. Special tokens and unknown IDs are skipped. */
  decode(ids) {
    let s = '';
    for (const id of ids) {
      if (id <= NovaTokenizer.EOS) continue;
      const t = this.itos[id];
      if (t !== undefined) s += t;
    }
    return s;
  }

  toJSON() {
    return { itos: this.itos.slice() };
  }

  static fromJSON(o) {
    const tk = new NovaTokenizer();
    tk.itos = (o && o.itos ? o.itos : []).slice();
    tk._rebuild();
    return tk;
  }
}
