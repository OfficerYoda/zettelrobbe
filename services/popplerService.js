// services/popplerService.js
//
// Poppler Service – renders PDF pages to PNG images by shelling out to the
// system `pdftoppm` binary (poppler-utils). Used by the OCR pipeline so that
// local vision models receive real page images instead of a single
// first-page thumbnail. Deliberately config-free: callers pass maxPages and
// dpi explicitly, which keeps this service trivially unit-testable.

const { execFile } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const os = require('os');
const path = require('path');

const PROBE_TIMEOUT_MS = 5000;
const DEFAULT_RENDER_TIMEOUT_MS = 120000;
// PNG magic number (first 8 bytes) and the trailing IEND chunk marker. A file
// that lacks either was never finished being written — pdftoppm was killed
// mid-render (e.g. hit the render timeout on a large/high-DPI page) — and must
// not be shipped to the OCR model, which rejects it with an HTTP 400.
const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);
const PNG_IEND = Buffer.from('IEND', 'latin1');
// pdftoppm writes images to files; stdout/stderr only carry diagnostics.
const MAX_STDIO_BUFFER_BYTES = 1024 * 1024;
// Defensive cap so a single rendered page cannot blow up request payloads.
const MAX_PAGE_FILE_BYTES = 32 * 1024 * 1024;

class PopplerService {
  constructor() {
    // Instance property so tests can inject a fake execFile implementation.
    this._execFile = promisify(execFile);
    this._availabilityPromise = null;
  }

  /**
   * Whether the system pdftoppm binary is usable. Memoized per process.
   * Only a missing/inaccessible binary counts as unavailable: poppler tools
   * print version info to stderr and the exit code of `-v` is not reliable
   * across poppler versions.
   * @returns {Promise<boolean>}
   */
  isAvailable() {
    if (!this._availabilityPromise) {
      this._availabilityPromise = this._probePdftoppm();
    }
    return this._availabilityPromise;
  }

  async _probePdftoppm() {
    try {
      await this._execFile('pdftoppm', ['-v'], { timeout: PROBE_TIMEOUT_MS });
      return true;
    } catch (error) {
      if (error && (error.code === 'ENOENT' || error.code === 'EACCES')) {
        return false;
      }
      const output = `${error?.stdout || ''}${error?.stderr || ''}`;
      return output.toLowerCase().includes('pdftoppm');
    }
  }

  /**
   * Whether a buffer holds a complete PNG file: the 8-byte signature at the
   * start and the IEND chunk marker near the end. pdftoppm writes files
   * incrementally, so a process killed mid-write leaves a valid-looking prefix
   * with no IEND — exactly the corruption that makes the OCR model return 400.
   * @param {Buffer} buffer
   * @returns {boolean}
   */
  isCompletePng(buffer) {
    if (!Buffer.isBuffer(buffer) || buffer.length < 16) {
      return false;
    }
    if (!buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
      return false;
    }
    // The IEND chunk is the last 12 bytes; search the tail generously.
    return buffer.subarray(-16).includes(PNG_IEND);
  }

  /**
   * Render the first pages of a PDF to PNG images via pdftoppm.
   * @param {Buffer} pdfBuffer - raw PDF bytes
   * @param {{maxPages: number, dpi: number, timeoutMs?: number}} options
   * @returns {Promise<{
   *   pages: Array<{base64: string, mimeType: string, pageNumber: number}>,
   *   totalPages: number|null,
   *   truncated: boolean
   * }>}
   */
  async renderPdfToImages(pdfBuffer, { maxPages, dpi, timeoutMs }) {
    const renderTimeoutMs =
      Number.isFinite(timeoutMs) && timeoutMs > 0
        ? timeoutMs
        : DEFAULT_RENDER_TIMEOUT_MS;
    const tempDir = await fs.mkdtemp(
      path.join(os.tmpdir(), 'paperless-ai-ocr-')
    );

    try {
      const inputPath = path.join(tempDir, 'input.pdf');
      const outputPrefix = path.join(tempDir, 'page');
      await fs.writeFile(inputPath, pdfBuffer);

      let renderError = null;
      try {
        await this._execFile(
          'pdftoppm',
          [
            '-png',
            '-r',
            String(dpi),
            '-f',
            '1',
            '-l',
            String(maxPages),
            inputPath,
            outputPrefix,
          ],
          {
            timeout: renderTimeoutMs,
            killSignal: 'SIGKILL',
            maxBuffer: MAX_STDIO_BUFFER_BYTES,
          }
        );
      } catch (error) {
        // Keep going: mildly corrupt PDFs often yield a non-zero exit code
        // after successfully rendering some pages.
        renderError = error;
      }

      const pageFiles = (await fs.readdir(tempDir))
        .map((name) => {
          const match = /^page-(\d+)\.png$/.exec(name);
          return match
            ? { name, pageNumber: Number.parseInt(match[1], 10) }
            : null;
        })
        .filter(Boolean)
        // pdftoppm zero-pads page numbers based on the document's total page
        // count, so numbers must be compared numerically, not lexically.
        .sort((a, b) => a.pageNumber - b.pageNumber);

      if (pageFiles.length === 0) {
        const detail = String(
          renderError?.stderr || renderError?.message || 'no output produced'
        ).trim();
        throw new Error(`pdftoppm failed to render any page: ${detail}`);
      }

      if (renderError) {
        const detail = String(
          renderError.stderr || renderError.message || ''
        ).trim();
        console.warn(
          `[OCR] pdftoppm exited with an error after rendering ${pageFiles.length} page(s), continuing with partial output: ${detail}`
        );
      }

      const pages = [];
      for (const { name, pageNumber } of pageFiles) {
        const fileBuffer = await fs.readFile(path.join(tempDir, name));
        if (fileBuffer.length > MAX_PAGE_FILE_BYTES) {
          console.warn(
            `[OCR] Skipping rendered page ${pageNumber}: image is ${fileBuffer.length} bytes (limit ${MAX_PAGE_FILE_BYTES})`
          );
          continue;
        }
        // A killed pdftoppm leaves the in-flight page half-written. Sending it
        // to the OCR model fails the whole document with an HTTP 400, so drop
        // any page whose PNG is incomplete rather than shipping garbage.
        if (!this.isCompletePng(fileBuffer)) {
          console.warn(
            `[OCR] Skipping rendered page ${pageNumber}: PNG is incomplete (pdftoppm was likely killed mid-render — raise OCR_PDF_RENDER_TIMEOUT_MS or lower OCR_PDF_RENDER_DPI)`
          );
          continue;
        }
        pages.push({
          base64: fileBuffer.toString('base64'),
          mimeType: 'image/png',
          pageNumber,
        });
      }

      if (pages.length === 0) {
        throw new Error(
          'pdftoppm produced no usable page images (all pages were oversized or incomplete)'
        );
      }

      // Only when the page limit was hit is the total page count interesting
      // (for an accurate truncation warning). pdfinfo is best-effort.
      let totalPages = null;
      if (pageFiles.length >= maxPages) {
        totalPages = await this._readTotalPages(inputPath);
      }
      const truncated =
        pageFiles.length >= maxPages &&
        (totalPages === null || totalPages > maxPages);

      return { pages, totalPages, truncated };
    } finally {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
    }
  }

  async _readTotalPages(inputPath) {
    try {
      const { stdout } = await this._execFile('pdfinfo', [inputPath], {
        timeout: PROBE_TIMEOUT_MS,
        maxBuffer: MAX_STDIO_BUFFER_BYTES,
      });
      const match = /^Pages:\s+(\d+)/m.exec(String(stdout));
      return match ? Number.parseInt(match[1], 10) : null;
    } catch {
      return null;
    }
  }
}

module.exports = new PopplerService();
