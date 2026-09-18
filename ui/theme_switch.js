// A sidebar control for Streamlit's own Light/Dark theme. Never recolor a
// canvas or modify React internals: select the same native menu item as a user.
// The menu test IDs are covered by the browser acceptance check (1.58/1.61).
export default function(component) {
    const { parentElement, data, setStateValue } = component;
    const button = parentElement.querySelector('button');
    const status = parentElement.querySelector('[role="status"]');
    let reported = data.mode;
    let busy = false;
    let disposed = false;
    let frame = 0;
    const currentMode = () => {
        const rgb = getComputedStyle(document.body).backgroundColor.match(/[\d.]+/g);
        return rgb && (Number(rgb[0])*0.2126 + Number(rgb[1])*0.7152 + Number(rgb[2])*0.0722) < 128
            ? 'dark' : 'light';
    };
    const sync = () => {
        if (disposed) return;
        const mode = currentMode();
        if (document.documentElement.dataset.quantTheme !== mode)
            document.documentElement.dataset.quantTheme = mode;
        button.setAttribute('aria-checked', String(mode === 'dark'));
        button.disabled = busy || data.disabled;
        if (!busy && mode !== reported) {
            reported = mode;
            setStateValue('mode', mode);
        }
    };
    const schedule = () => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(sync);
    };
    const waitFor = async (selector) => {
        for (let attempt = 0; attempt < 40 && !disposed; attempt++) {
            const element = document.querySelector(selector);
            if (element) return element;
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        throw new Error('Native theme menu unavailable');
    };
    button.onclick = async () => {
        if (busy) return;
        busy = true;
        button.disabled = true;
        status.textContent = '';
        const target = currentMode() === 'dark' ? 'Light' : 'Dark';
        let opened = false;
        try {
            const menuButton = document.querySelector('[data-testid="stMainMenu"] button, button[aria-label="Main menu"]');
            if (!menuButton) throw new Error('Native theme menu unavailable');
            if (menuButton.getAttribute('aria-expanded') !== 'true') {
                menuButton.click();
                opened = true;
            }
            const option = await waitFor(`[data-testid="stMainMenuItem-theme-${target}"]`);
            option.click();
            for (let i = 0; i < 40 && currentMode() !== target.toLowerCase(); i++)
                await new Promise(resolve => setTimeout(resolve, 50));
            if (currentMode() !== target.toLowerCase()) throw new Error('Theme did not change');
        } catch (error) {
            status.textContent = 'Open ⋮ and choose Light or Dark under Theme.';
            console.error('Quant theme switch:', error);
        } finally {
            const menuButton = document.querySelector('[data-testid="stMainMenu"] button, button[aria-label="Main menu"]');
            if (opened && menuButton?.getAttribute('aria-expanded') === 'true') menuButton.click();
            busy = false;
            sync();
            button.focus({preventScroll: true});
        }
    };
    // Native menu changes, system preference changes, and route navigation all
    // update body/app classes. Observe those without polling or page reloads.
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, {attributes: true});
    const app = document.querySelector('[data-testid="stApp"]');
    if (app) observer.observe(app, {attributes: true});
    observer.observe(document.head, {childList: true, subtree: true});
    sync();
    return () => {
        disposed = true;
        observer.disconnect();
        cancelAnimationFrame(frame);
        button.onclick = null;
    };
}
