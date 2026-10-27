/**
 * LiteSTEP file detection.
 *
 * Uses the same regex-based heuristics as the existing v0.4.0 extension
 * to identify Python files that contain Lite-STEP building DSL code.
 */

const PATTERNS: RegExp[] = [
  /from lite_step\.models import/,
  /from lite_step import/,
  /Building\s*\(/,
  /bldg\.add\s*\(/,
  /result\s*=\s*(?:bldg|building)/i,
  /generate_building\s*\(/,
  /Solid\s*\(/,
  /Wall\s*\(/,
  /Point\s*\(x=/,
  /from model_sketching/,
];

const MIN_MATCHES = 2;

/** Cache: filePath → isLiteStep */
const _cache: Map<string, boolean> = new Map();

/**
 * Check whether a Python file's text content matches Lite-STEP patterns.
 * Results are cached by file path until clearCache() is called.
 */
export function isLiteStepFile(text: string, filePath?: string): boolean {
  if (filePath && _cache.has(filePath)) {
    return _cache.get(filePath)!;
  }

  let matchCount = 0;
  for (const pattern of PATTERNS) {
    if (pattern.test(text)) {
      matchCount++;
      if (matchCount >= MIN_MATCHES) {
        if (filePath) {
          _cache.set(filePath, true);
        }
        return true;
      }
    }
  }

  if (filePath) {
    _cache.set(filePath, false);
  }
  return false;
}

/**
 * Clear the detection cache.
 * Called when the user triggers a refresh or a file is modified.
 */
export function clearCache(): void {
  _cache.clear();
}
