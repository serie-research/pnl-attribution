async function activate(host) {
  if (host.dataset.activating) return;
  host.dataset.activating = 'true';
  let readinessObserver;
  const reveal = error => {
    clearTimeout(deadline);
    readinessObserver?.disconnect();
    if (error) host.dataset.embedError = 'true';
    host.dataset.activated = 'true';
    host.dispatchEvent(new Event('embed-ready', { bubbles: true }));
  };
  const deadline = setTimeout(() => {
    let status = host.querySelector('[role="status"], .status');
    if (!status) {
      status = document.createElement('p');
      status.setAttribute('role', 'status');
      host.append(status);
    }
    status.textContent = 'Figure unavailable: loading timed out. Reload to try again.';
    reveal(true);
  }, 30_000);
  try {
    if (!window.d3) throw new Error('Could not load figure library');
    const template = host.querySelector('template');
    const fragment = template.content.cloneNode(true);
    // Remove scripts before insertion: parser-created template scripts can run
    // when connected. Insert fresh scripts exactly once, in document order.
    const scripts = [...fragment.querySelectorAll('script')].map(inert => {
      const marker = document.createComment('embed script');
      inert.replaceWith(marker);
      return { inert, marker };
    });
    host.append(fragment);
    template.remove();
    for (const { inert, marker } of scripts) {
      const script = document.createElement('script');
      for (const attr of inert.attributes) script.setAttribute(attr.name, attr.value);
      script.textContent = inert.textContent;
      const loaded = script.src ? new Promise((resolve, reject) => { script.onload = resolve; script.onerror = reject; }) : null;
      marker.replaceWith(script);
      if (loaded) await loaded;
    }
    // Hold the reserved space through the asynchronous data fetch and first draw.
    const drawn = () => !!host.querySelector('svg :is(path,rect,circle,line,text)');
    const failed = () => [...host.querySelectorAll('[role="status"], .status')]
      .some(node => /^(Figure unavailable|Could not load)/.test(node.textContent.trim()));
    const ready = () => reveal(failed());
    if (drawn() || failed()) ready();
    else {
      readinessObserver = new MutationObserver(() => { if (drawn() || failed()) ready(); });
      readinessObserver.observe(host, { childList: true, subtree: true, characterData: true });
    }
  } catch (error) {
    host.textContent = `Figure unavailable: ${error.message}`;
    reveal(true);
    console.error(error);
  }
}
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    observer.unobserve(entry.target);
    activate(entry.target);
  }
}, { rootMargin: '100% 0px' });
function reserve(host) {
  const sizes = JSON.parse(host.dataset.embedSizes);
  const right = sizes.findIndex(([width]) => width >= innerWidth);
  const [w0, h0] = sizes[Math.max(0, right - 1)];
  const [w1, h1] = sizes[right < 0 ? sizes.length - 1 : right];
  const height = right < 0 ? h1 : w1 === w0 ? h0 : h0 + (h1 - h0) * (innerWidth - w0) / (w1 - w0);
  host.style.setProperty('--embed-placeholder-height', `${height}px`);
}
window.observeArticleEmbed = host => {
  reserve(host);
  const rect = host.getBoundingClientRect();
  if (rect.height && rect.bottom >= -innerHeight && rect.top <= innerHeight * 2) activate(host);
  else observer.observe(host);
};
window.addEventListener('resize', () => {
  for (const host of document.querySelectorAll('[data-lazy-embed]:not([data-activated])')) reserve(host);
});
