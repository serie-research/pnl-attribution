import notebook from '../../../notebooks/signal-attribution.ipynb?raw';
export function GET() {
  return new Response(notebook, { headers: { 'Content-Type': 'application/json' } });
}
