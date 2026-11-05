// Light/dark theme button shared by the article and the presentation view.
// An explicit choice is stored; otherwise the page follows the system setting.
// ThemeInit.astro applies the stored or system theme before first paint.

const KEY = 'theme';

function stored() {
  try { return localStorage.getItem(KEY); } catch { return null; }
}

export function themeToggle(button, onChange = () => {}) {
  const root = document.documentElement;
  const label = () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    button.setAttribute('aria-label', `Switch to ${next} theme`);
    button.title = `Switch to ${next} theme`;
  };
  const apply = dark => {
    if (dark) root.dataset.theme = 'dark'; else delete root.dataset.theme;
    label();
    onChange();
  };
  button.addEventListener('click', () => {
    const dark = root.dataset.theme !== 'dark';
    try { localStorage.setItem(KEY, dark ? 'dark' : 'light'); } catch {}
    apply(dark);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (!stored()) apply(e.matches);
  });
  label();
}
