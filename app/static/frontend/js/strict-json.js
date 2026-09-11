// A JSON grammar scanner runs before JSON.parse so duplicate keys cannot disappear.
export function parseStrictJSON(text) {
  let i = 0;
  const fail = () => { throw new Error(`Invalid strict JSON near character ${i + 1}. Check syntax, duplicate keys, finite numbers, and nesting (maximum 8).`); };
  const space = () => { while (/[\x20\t\r\n]/.test(text[i] ?? '\0')) i++; };
  function string() {
    const start = i++;
    while (i < text.length) {
      if (text[i] === '"') {
        i++;
        try { return JSON.parse(text.slice(start, i)); } catch { fail(); }
      }
      if (text[i] === '\\') i++;
      i++;
    }
    fail();
  }
  function value(depth = 0) {
    space();
    if (text[i] === '{' || text[i] === '[') {
      if (depth >= 8) fail();
      const object = text[i++] === '{';
      const close = object ? '}' : ']';
      const keys = new Set();
      space();
      if (text[i] === close) { i++; return; }
      while (i < text.length) {
        if (object) {
          space();
          if (text[i] !== '"') fail();
          const key = string();
          if (keys.has(key)) fail();
          keys.add(key);
          space();
          if (text[i++] !== ':') fail();
        }
        value(depth + 1);
        space();
        if (text[i] === close) { i++; return; }
        if (text[i++] !== ',') fail();
      }
      fail();
    }
    if (text[i] === '"') { string(); return; }
    for (const literal of ['true', 'false', 'null']) {
      if (text.startsWith(literal, i)) { i += literal.length; return; }
    }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(i));
    if (!match || !Number.isFinite(Number(match[0]))) fail();
    i += match[0].length;
  }
  value();
  space();
  if (i !== text.length) fail();
  try { return JSON.parse(text); } catch { fail(); }
}
