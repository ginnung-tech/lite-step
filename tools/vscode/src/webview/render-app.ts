/**
 * Webview render app placeholder.
 *
 * The actual Three.js rendering code is inlined in preview-panel.ts's HTML
 * to avoid complex bundling of Three.js + web-ifc WASM for the webview.
 *
 * This file exists as the esbuild webview entry point. In a future version,
 * the ThatOpen/web-ifc rendering stack can be bundled here for full IFC
 * parsing support.
 */

export {};
